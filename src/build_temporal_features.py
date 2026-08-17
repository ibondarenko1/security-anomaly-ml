"""Build leakage-safe temporal context features for CIC-UNSW-NB15 v2.

The pipeline performs an external timestamp sort, processes each capture day as
an independent split, and excludes the complete current timestamp peer group
from every history window.  Raw flow/host identifiers and timestamps are used
only to build context and are not written to the model-ready Parquet files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cic_unsw_nb15" / "CICFlowMeter_out.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
MANIFEST_PATH = OUTPUT_DIR / "feature_manifest.json"
BUILD_DATABASE_PATH = OUTPUT_DIR / "_temporal_feature_build.duckdb"
TEMP_DIRECTORY = OUTPUT_DIR / "_duckdb_temp"

SPLITS = {
    "train": {
        "capture_date": "2015-01-22",
        "expected_rows": 1_765_922,
        "filename": "train_features.parquet",
    },
    "validation": {
        "capture_date": "2015-02-17",
        "expected_rows": 498_890,
        "filename": "validation_features.parquet",
    },
    "locked_holdout": {
        "capture_date": "2015-02-18",
        "expected_rows": 1_275_429,
        "filename": "locked_holdout_features.parquet",
    },
}

RAW_IDENTIFIER_COLUMNS = ("Flow ID", "Src IP", "Dst IP", "Timestamp")
CONTEXT_KEYS = ("Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol")
NON_BASELINE_COLUMNS = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
)
TARGET_COLUMNS = ("attack_cat", "label")

STATIC_BEHAVIORAL_FEATURES = (
    "Src Port",
    "Dst Port",
    "Protocol",
    "src_port_well_known",
    "src_port_registered",
    "src_port_ephemeral",
    "dst_port_well_known",
    "dst_port_registered",
    "dst_port_ephemeral",
)

CONTEXT_FEATURES = (
    "src_conn_10s",
    "src_conn_60s",
    "src_unique_dst_10s",
    "src_unique_dst_60s",
    "src_unique_dport_10s",
    "src_unique_dport_60s",
    "src_dst_conn_10s",
    "src_dst_conn_60s",
    "src_dport_conn_10s",
    "src_dport_conn_60s",
    "src_bytes_10s",
    "src_bytes_60s",
    "src_packets_10s",
    "src_packets_60s",
    "src_mean_bytes_per_flow_60s",
    "src_mean_packets_per_flow_60s",
    "seconds_since_src_last_flow",
    "seconds_since_src_dst_last_flow",
    "src_tcp_ratio_60s",
    "src_udp_ratio_60s",
    "src_syn_count_60s",
    "src_rst_count_60s",
    "src_dport_diversity_60s",
    "dst_conn_10s",
    "dst_conn_60s",
    "dst_unique_src_10s",
    "dst_unique_src_60s",
    "dst_unique_sport_10s",
    "dst_unique_sport_60s",
    "dst_sport_conn_10s",
    "dst_sport_conn_60s",
    "dst_bytes_10s",
    "dst_bytes_60s",
    "dst_packets_10s",
    "dst_packets_60s",
    "dst_mean_bytes_per_flow_60s",
    "dst_mean_packets_per_flow_60s",
    "seconds_since_dst_last_flow",
    "dst_tcp_ratio_60s",
    "dst_udp_ratio_60s",
    "dst_syn_count_60s",
    "dst_rst_count_60s",
    "dst_sport_diversity_60s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit; external operators spill to disk (default: 8GB).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="DuckDB worker threads (default: up to 8).",
    )
    parser.add_argument(
        "--keep-build-database",
        action="store_true",
        help="Retain the temporary DuckDB staging database after a successful build.",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    missing = [column for column in NON_BASELINE_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    return columns


def baseline_features(raw_columns: list[str]) -> list[str]:
    excluded = set(NON_BASELINE_COLUMNS)
    features = [column for column in raw_columns if column not in excluded]
    if len(features) != 76:
        raise ValueError(f"Expected 76 CICFlowMeter features, found {len(features)}")
    return features


def window_clause() -> str:
    """Return named tie-safe windows shared by all feature expressions."""
    return """
WINDOW
    src10 AS (
        PARTITION BY "Src IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src60 AS (
        PARTITION BY "Src IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_history AS (
        PARTITION BY "Src IP" ORDER BY event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst10 AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst60 AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst_history AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dport10 AS (
        PARTITION BY "Src IP", "Dst Port" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dport60 AS (
        PARTITION BY "Src IP", "Dst Port" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst10 AS (
        PARTITION BY "Dst IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst60 AS (
        PARTITION BY "Dst IP" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_history AS (
        PARTITION BY "Dst IP" ORDER BY event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_sport10 AS (
        PARTITION BY "Dst IP", "Src Port" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_sport60 AS (
        PARTITION BY "Dst IP", "Src Port" ORDER BY event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    )
""".strip()


def context_expressions() -> list[str]:
    """Return SQL expressions implementing the frozen v2 context contract."""
    return [
        "(count(*) OVER src10)::BIGINT AS src_conn_10s",
        "(count(*) OVER src60)::BIGINT AS src_conn_60s",
        '(count(DISTINCT "Dst IP") OVER src10)::BIGINT AS src_unique_dst_10s',
        '(count(DISTINCT "Dst IP") OVER src60)::BIGINT AS src_unique_dst_60s',
        '(count(DISTINCT "Dst Port") OVER src10)::BIGINT AS src_unique_dport_10s',
        '(count(DISTINCT "Dst Port") OVER src60)::BIGINT AS src_unique_dport_60s',
        "(count(*) OVER src_dst10)::BIGINT AS src_dst_conn_10s",
        "(count(*) OVER src_dst60)::BIGINT AS src_dst_conn_60s",
        "(count(*) OVER src_dport10)::BIGINT AS src_dport_conn_10s",
        "(count(*) OVER src_dport60)::BIGINT AS src_dport_conn_60s",
        "coalesce(sum(flow_bytes) OVER src10, 0.0) AS src_bytes_10s",
        "coalesce(sum(flow_bytes) OVER src60, 0.0) AS src_bytes_60s",
        "coalesce(sum(flow_packets) OVER src10, 0.0) AS src_packets_10s",
        "coalesce(sum(flow_packets) OVER src60, 0.0) AS src_packets_60s",
        "coalesce(avg(flow_bytes) OVER src60, 0.0) AS src_mean_bytes_per_flow_60s",
        "coalesce(avg(flow_packets) OVER src60, 0.0) AS src_mean_packets_per_flow_60s",
        "coalesce(date_diff('second', max(event_ts) OVER src_history, event_ts), -1)::BIGINT AS seconds_since_src_last_flow",
        "coalesce(date_diff('second', max(event_ts) OVER src_dst_history, event_ts), -1)::BIGINT AS seconds_since_src_dst_last_flow",
        "coalesce(sum(CASE WHEN \"Protocol\" = 6 THEN 1 ELSE 0 END) OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_tcp_ratio_60s",
        "coalesce(sum(CASE WHEN \"Protocol\" = 17 THEN 1 ELSE 0 END) OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_udp_ratio_60s",
        'coalesce(sum("SYN Flag Count") OVER src60, 0)::DOUBLE AS src_syn_count_60s',
        'coalesce(sum("RST Flag Count") OVER src60, 0)::DOUBLE AS src_rst_count_60s',
        'coalesce(count(DISTINCT "Dst Port") OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_dport_diversity_60s',
        "(count(*) OVER dst10)::BIGINT AS dst_conn_10s",
        "(count(*) OVER dst60)::BIGINT AS dst_conn_60s",
        '(count(DISTINCT "Src IP") OVER dst10)::BIGINT AS dst_unique_src_10s',
        '(count(DISTINCT "Src IP") OVER dst60)::BIGINT AS dst_unique_src_60s',
        '(count(DISTINCT "Src Port") OVER dst10)::BIGINT AS dst_unique_sport_10s',
        '(count(DISTINCT "Src Port") OVER dst60)::BIGINT AS dst_unique_sport_60s',
        "(count(*) OVER dst_sport10)::BIGINT AS dst_sport_conn_10s",
        "(count(*) OVER dst_sport60)::BIGINT AS dst_sport_conn_60s",
        "coalesce(sum(flow_bytes) OVER dst10, 0.0) AS dst_bytes_10s",
        "coalesce(sum(flow_bytes) OVER dst60, 0.0) AS dst_bytes_60s",
        "coalesce(sum(flow_packets) OVER dst10, 0.0) AS dst_packets_10s",
        "coalesce(sum(flow_packets) OVER dst60, 0.0) AS dst_packets_60s",
        "coalesce(avg(flow_bytes) OVER dst60, 0.0) AS dst_mean_bytes_per_flow_60s",
        "coalesce(avg(flow_packets) OVER dst60, 0.0) AS dst_mean_packets_per_flow_60s",
        "coalesce(date_diff('second', max(event_ts) OVER dst_history, event_ts), -1)::BIGINT AS seconds_since_dst_last_flow",
        "coalesce(sum(CASE WHEN \"Protocol\" = 6 THEN 1 ELSE 0 END) OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_tcp_ratio_60s",
        "coalesce(sum(CASE WHEN \"Protocol\" = 17 THEN 1 ELSE 0 END) OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_udp_ratio_60s",
        'coalesce(sum("SYN Flag Count") OVER dst60, 0)::DOUBLE AS dst_syn_count_60s',
        'coalesce(sum("RST Flag Count") OVER dst60, 0)::DOUBLE AS dst_rst_count_60s',
        'coalesce(count(DISTINCT "Src Port") OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_sport_diversity_60s',
    ]


def enriched_query(source_relation: str, baseline: list[str]) -> str:
    baseline_sql = ",\n        ".join(quote_identifier(name) for name in baseline)
    context_sql = ",\n        ".join(context_expressions())
    return f"""
SELECT
        event_ts,
        {baseline_sql},
        "Src Port",
        "Dst Port",
        "Protocol",
        CASE WHEN "Src Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT AS src_port_well_known,
        CASE WHEN "Src Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT AS src_port_registered,
        CASE WHEN "Src Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT AS src_port_ephemeral,
        CASE WHEN "Dst Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT AS dst_port_well_known,
        CASE WHEN "Dst Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT AS dst_port_registered,
        CASE WHEN "Dst Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT AS dst_port_ephemeral,
        {context_sql},
        "Label" AS attack_cat,
        CASE WHEN "Label" = 'Benign' THEN 0 ELSE 1 END::UTINYINT AS label
FROM {source_relation}
{window_clause()}
""".strip()


def output_query(source_relation: str, baseline: list[str]) -> str:
    output_columns = [
        *baseline,
        *STATIC_BEHAVIORAL_FEATURES,
        *CONTEXT_FEATURES,
        *TARGET_COLUMNS,
    ]
    selected = ",\n    ".join(quote_identifier(name) for name in output_columns)
    return f"""
WITH enriched AS (
{enriched_query(source_relation, baseline)}
)
SELECT
    {selected}
FROM enriched
ORDER BY event_ts
""".strip()


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    memory_limit: str,
    threads: int,
) -> None:
    connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_string(TEMP_DIRECTORY)}")


def create_staging_table(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    started = time.perf_counter()
    connection.execute("DROP TABLE IF EXISTS flows")
    connection.execute(
        f"""
        CREATE TABLE flows AS
        SELECT
            * EXCLUDE ("Flow ID", "Timestamp"),
            strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
            ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE AS flow_bytes,
            ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
        FROM read_csv_auto(
            {sql_string(RAW_PATH)},
            header = true,
            sample_size = -1,
            parallel = true
        )
        ORDER BY event_ts
        """
    )
    stats = connection.execute(
        """
        SELECT
            count(*) AS rows,
            min(event_ts) AS minimum_timestamp,
            max(event_ts) AS maximum_timestamp,
            count(DISTINCT CAST(event_ts AS DATE)) AS capture_days
        FROM flows
        """
    ).fetchone()
    if stats is None:
        raise RuntimeError("Staging table validation returned no result")
    return {
        "rows": int(stats[0]),
        "minimum_timestamp": stats[1].isoformat(),
        "maximum_timestamp": stats[2].isoformat(),
        "capture_days": int(stats[3]),
        "runtime_seconds": time.perf_counter() - started,
    }


def validate_output(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    expected_rows: int,
    expected_columns: list[str],
) -> dict[str, Any]:
    describe_rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(path)})"
    ).fetchall()
    columns = [row[0] for row in describe_rows]
    if columns != expected_columns:
        raise AssertionError("Generated Parquet schema does not match feature contract")
    forbidden = sorted(set(RAW_IDENTIFIER_COLUMNS) & set(columns))
    if forbidden:
        raise AssertionError(f"Raw identifiers leaked into output: {forbidden}")

    stats = connection.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(*) FILTER (WHERE label NOT IN (0, 1) OR label IS NULL) AS bad_targets,
            count(*) FILTER (WHERE attack_cat IS NULL) AS missing_attack_categories
        FROM read_parquet({sql_string(path)})
        """
    ).fetchone()
    if stats is None:
        raise RuntimeError("Output validation returned no result")
    rows, bad_targets, missing_categories = map(int, stats)
    if rows != expected_rows:
        raise AssertionError(f"Expected {expected_rows:,} rows, found {rows:,}")
    if bad_targets or missing_categories:
        raise AssertionError(
            f"Invalid targets: bad binary={bad_targets}, missing category={missing_categories}"
        )

    context_checks = " + ".join(
        f"count(*) FILTER (WHERE {quote_identifier(column)} IS NULL "
        f"OR NOT isfinite({quote_identifier(column)}))"
        for column in CONTEXT_FEATURES
    )
    invalid_context = int(
        connection.execute(
            f"SELECT {context_checks} FROM read_parquet({sql_string(path)})"
        ).fetchone()[0]
    )
    if invalid_context:
        raise AssertionError(f"Context output contains {invalid_context} invalid values")

    target_distribution = connection.execute(
        f"""
        SELECT attack_cat, label, count(*)
        FROM read_parquet({sql_string(path)})
        GROUP BY attack_cat, label
        ORDER BY attack_cat
        """
    ).fetchall()
    return {
        "rows": rows,
        "columns": len(columns),
        "raw_identifier_columns_present": forbidden,
        "invalid_context_values": invalid_context,
        "target_distribution": [
            {"attack_cat": category, "label": int(label), "count": int(count)}
            for category, label, count in target_distribution
        ],
    }


def build_split(
    connection: duckdb.DuckDBPyConnection,
    split_name: str,
    split: dict[str, Any],
    baseline: list[str],
) -> dict[str, Any]:
    capture_date = split["capture_date"]
    final_path = OUTPUT_DIR / split["filename"]
    temporary_path = final_path.with_suffix(".tmp.parquet")
    if temporary_path.exists():
        temporary_path.unlink()

    connection.execute("DROP VIEW IF EXISTS current_split")
    connection.execute(
        f"""
        CREATE TEMP VIEW current_split AS
        SELECT *
        FROM flows
        WHERE CAST(event_ts AS DATE) = DATE {sql_string(capture_date)}
        """
    )
    actual_input_rows = int(
        connection.execute("SELECT count(*) FROM current_split").fetchone()[0]
    )
    if actual_input_rows != split["expected_rows"]:
        raise AssertionError(
            f"{split_name}: expected {split['expected_rows']:,} input rows, "
            f"found {actual_input_rows:,}"
        )

    started = time.perf_counter()
    query = output_query("current_split", baseline)
    connection.execute(
        f"""
        COPY ({query})
        TO {sql_string(temporary_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    build_seconds = time.perf_counter() - started

    expected_columns = [
        *baseline,
        *STATIC_BEHAVIORAL_FEATURES,
        *CONTEXT_FEATURES,
        *TARGET_COLUMNS,
    ]
    validation = validate_output(
        connection,
        temporary_path,
        split["expected_rows"],
        expected_columns,
    )
    os.replace(temporary_path, final_path)
    validation.update(
        {
            "capture_date": capture_date,
            "path": str(final_path.relative_to(PROJECT_ROOT)),
            "size_bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
            "build_seconds": build_seconds,
            "context_state_initialized_empty": True,
            "context_state_received_prior_split_rows": False,
        }
    )
    return validation


def cleanup_build_files() -> None:
    for path in (
        BUILD_DATABASE_PATH,
        BUILD_DATABASE_PATH.with_suffix(BUILD_DATABASE_PATH.suffix + ".wal"),
    ):
        if path.is_file():
            path.unlink()
    if TEMP_DIRECTORY.is_dir():
        try:
            TEMP_DIRECTORY.rmdir()
        except OSError:
            # DuckDB can leave an empty database-specific subdirectory on some
            # platforms. It is safe to retain generated spill metadata.
            pass


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing CIC-UNSW-NB15 full export: {RAW_PATH}")

    raw_columns = read_raw_columns(RAW_PATH)
    baseline = baseline_features(raw_columns)
    if set(RAW_IDENTIFIER_COLUMNS) & set(baseline):
        raise AssertionError("A raw identifier was included in baseline features")
    if len(context_expressions()) != len(CONTEXT_FEATURES):
        raise AssertionError("Context SQL and context feature contract differ")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = duckdb.connect(str(BUILD_DATABASE_PATH))
    try:
        configure_connection(connection, args.memory_limit, args.threads)
        print("Creating timestamp-sorted staging table...", flush=True)
        staging = create_staging_table(connection)
        if staging["rows"] != sum(split["expected_rows"] for split in SPLITS.values()):
            raise AssertionError("Staging row count does not match frozen split totals")
        if staging["capture_days"] != len(SPLITS):
            raise AssertionError("Unexpected number of capture days")
        print(
            f"Staged {staging['rows']:,} rows in {staging['runtime_seconds']:.1f}s",
            flush=True,
        )

        split_results: dict[str, Any] = {}
        for split_name, split in SPLITS.items():
            print(
                f"Building {split_name} from {split['capture_date']} with empty state...",
                flush=True,
            )
            split_results[split_name] = build_split(
                connection,
                split_name,
                split,
                baseline,
            )
            print(
                f"Built {split_name}: {split_results[split_name]['rows']:,} rows "
                f"in {split_results[split_name]['build_seconds']:.1f}s",
                flush=True,
            )
    finally:
        connection.close()

    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Security Anomaly ML v2 - Temporal Context Network Triage",
        "source": {
            "path": str(RAW_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(RAW_PATH),
            "rows": staging["rows"],
        },
        "split_policy": {
            "train": "2015-01-22",
            "validation": "2015-02-17",
            "locked_holdout": "2015-02-18",
            "sort_key": ["Timestamp"],
            "same_timestamp_policy": (
                "All flows sharing timestamp t are peers. Features use only "
                "timestamps < t; the complete peer group is excluded."
            ),
            "context_state_reset_at_each_split": True,
            "host_overlap_allowed": True,
            "raw_identifiers_used_as_model_features": False,
            "locked_holdout_policy": (
                "No threshold, feature, or hyperparameter changes may be selected "
                "from locked-holdout results."
            ),
        },
        "feature_contract": {
            "baseline_v2_features": baseline,
            "baseline_v2_feature_count": len(baseline),
            "context_v2_static_behavioral_features": list(STATIC_BEHAVIORAL_FEATURES),
            "context_v2_temporal_features": list(CONTEXT_FEATURES),
            "context_v2_feature_count": (
                len(baseline)
                + len(STATIC_BEHAVIORAL_FEATURES)
                + len(CONTEXT_FEATURES)
            ),
            "targets": list(TARGET_COLUMNS),
            "excluded_raw_identifiers": list(RAW_IDENTIFIER_COLUMNS),
            "flow_bytes_definition": (
                "Total Length of Fwd Packet + Total Length of Bwd Packet"
            ),
            "flow_packets_definition": "Total Fwd Packet + Total Bwd packets",
            "cold_start_seconds_since_value": -1,
            "port_ranges": {
                "well_known": "0-1023",
                "registered": "1024-49151",
                "ephemeral": "49152-65535",
            },
        },
        "outputs": split_results,
        "build": {
            "duckdb_version": duckdb.__version__,
            "memory_limit": args.memory_limit,
            "threads": args.threads,
            "total_runtime_seconds": time.perf_counter() - started,
            "model_trained": False,
            "threshold_selected": False,
            "locked_holdout_evaluated": False,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.keep_build_database:
        cleanup_build_files()

    print(f"Feature manifest: {MANIFEST_PATH}")
    print("No model was trained and the locked holdout was not evaluated.")


if __name__ == "__main__":
    main()
