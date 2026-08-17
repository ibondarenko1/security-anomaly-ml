"""Measure incident-level alert volume for the frozen CIC Context detector.

This stage is deliberately evaluation-only.  It uses the frozen February 17
Context scores at their frozen 0.10 threshold, restores raw identities solely
for grouping, and never fits or loads a classifier.  The combined source CSV is
scanned by timestamp, but only February 17 rows are materialized or parsed by
DuckDB; rows from every other date are discarded as opaque bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import sklearn

try:
    from .build_temporal_features import context_expressions, window_clause
except ImportError:  # Direct execution: python src/analyze_incident_aggregation.py
    from build_temporal_features import context_expressions, window_clause


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cic_unsw_nb15" / "CICFlowMeter_out.csv"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
VALIDATION_FEATURES_PATH = FEATURE_DIR / "validation_features.parquet"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"
CACHE_DIR = FEATURE_DIR / "_incident_aggregation_cache"
ROUTED_VALIDATION_CSV = CACHE_DIR / "feb17_routed_rows.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"
ABLATION_METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"

RESULTS_PATH = MODELS_DIR / "v2_incident_aggregation_results.csv"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_incident_aggregation_category_metrics.csv"
FP_PATTERNS_PATH = MODELS_DIR / "v2_incident_aggregation_fp_patterns.csv"
MANIFEST_PATH = MODELS_DIR / "v2_incident_aggregation_manifest.json"

CAPTURE_DATE = "2015-02-17"
CAPTURE_DATE_BYTES = b"17/02/2015"
EXPECTED_VALIDATION_ROWS = 498_890
FROZEN_THRESHOLD = 0.10
NEAR_PERFECT_INCIDENT_RECALL = 0.995

POLICIES: dict[str, tuple[str, ...]] = {
    "A": ("src_ip", "dst_ip"),
    "B": ("src_ip", "dst_ip", "dst_port"),
    "C": ("src_ip", "dst_ip", "dst_port", "protocol"),
}
POLICY_DESCRIPTIONS = {
    "A": "src_ip + dst_ip",
    "B": "src_ip + dst_ip + dst_port",
    "C": "src_ip + dst_ip + dst_port + protocol",
}
WINDOWS_SECONDS = (30, 60, 300, 900)
WINDOW_LABELS = {30: "30 seconds", 60: "60 seconds", 300: "5 minutes", 900: "15 minutes"}
ATTACK_CATEGORIES = (
    "Fuzzers",
    "Analysis",
    "Exploits",
    "DoS",
    "Reconnaissance",
    "Generic",
    "Backdoor",
    "Shellcode",
    "Worms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit for rebuilding February 17 context (default: 8GB).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="DuckDB worker threads (default: up to 8).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep the routed February 17 CSV after successful analysis.",
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


def locate_csv_field(line: bytes, field_index: int) -> bytes:
    """Return an unquoted early CSV field without parsing the remainder.

    The seven CIC identity fields precede all numeric flow features and contain
    no commas.  This lets the routing pass inspect only Timestamp (field 6).
    """

    start = 0
    for _ in range(field_index):
        comma = line.find(b",", start)
        if comma < 0:
            raise ValueError("Malformed CIC CSV row before Timestamp")
        start = comma + 1
    end = line.find(b",", start)
    if end < 0:
        raise ValueError("Malformed CIC CSV row at Timestamp")
    return line[start:end]


def route_validation_rows(
    raw_path: Path,
    output_path: Path,
    expected_rows: int = EXPECTED_VALIDATION_ROWS,
) -> dict[str, Any]:
    """Materialize only Feb 17 rows from the combined, unsorted raw CSV.

    Non-matching rows remain opaque bytes: only their Timestamp field is
    inspected to make the routing decision.  Their features and labels are not
    parsed, retained, counted by class, or exposed to the analysis.
    """

    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    if temporary.exists():
        temporary.unlink()

    digest = hashlib.sha256()
    matched_rows = 0
    total_lines_routed = 0
    with raw_path.open("rb", buffering=8 * 1024 * 1024) as source, temporary.open(
        "wb", buffering=8 * 1024 * 1024
    ) as target:
        header = source.readline()
        if not header:
            raise ValueError("Raw CIC CSV is empty")
        header_columns = header.decode("utf-8-sig").rstrip("\r\n").split(",")
        if len(header_columns) != 84 or header_columns[6] != "Timestamp":
            raise AssertionError("Unexpected CIC CSV schema or Timestamp position")
        target.write(header)
        digest.update(header)

        for line in source:
            total_lines_routed += 1
            timestamp = locate_csv_field(line, 6)
            if timestamp.startswith(CAPTURE_DATE_BYTES):
                target.write(line)
                digest.update(line)
                matched_rows += 1

    if matched_rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise AssertionError(
            f"Expected {expected_rows:,} February 17 rows, routed {matched_rows:,}"
        )
    os.replace(temporary, output_path)
    return {
        "rows": matched_rows,
        "combined_source_rows_scanned": total_lines_routed,
        "materialized_non_validation_rows": 0,
        "sha256": digest.hexdigest(),
        "size_bytes": output_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
        "routing_field": "Timestamp only",
    }


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    memory_limit: str,
    threads: int,
) -> None:
    connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_string(CACHE_DIR)}")


def feature_signature_expression(features: Iterable[str]) -> str:
    columns = ", ".join(quote_identifier(column) for column in features)
    return f"hash({columns})"


def enriched_identity_query(source_relation: str, baseline: list[str]) -> str:
    baseline_sql = ",\n        ".join(quote_identifier(name) for name in baseline)
    context_sql = ",\n        ".join(context_expressions())
    return f"""
SELECT
        event_ts AS timestamp,
        "Flow ID" AS flow_id,
        "Src IP" AS src_ip,
        "Dst IP" AS dst_ip,
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


def restore_validation_identities(
    routed_csv: Path,
    context_features: list[str],
    baseline_features: list[str],
    memory_limit: str,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align raw identities to frozen rows using the complete 128-feature input."""

    started = time.perf_counter()
    connection = duckdb.connect()
    try:
        configure_connection(connection, memory_limit, threads)
        connection.execute(
            f"""
            CREATE TABLE routed_validation AS
            SELECT
                *,
                strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
                ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE AS flow_bytes,
                ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
            FROM read_csv_auto(
                {sql_string(routed_csv)},
                header = true,
                sample_size = -1,
                parallel = true
            )
            """
        )
        routed_stats = connection.execute(
            """
            SELECT count(*), min(event_ts), max(event_ts),
                   count(*) FILTER (WHERE CAST(event_ts AS DATE) != DATE '2015-02-17')
            FROM routed_validation
            """
        ).fetchone()
        if routed_stats is None or int(routed_stats[0]) != EXPECTED_VALIDATION_ROWS:
            raise AssertionError("Routed February 17 row count changed")
        if int(routed_stats[3]) != 0:
            raise AssertionError("A non-February 17 row entered the identity table")

        connection.execute(
            f"CREATE TABLE enriched_identity AS {enriched_identity_query('routed_validation', baseline_features)}"
        )
        signature = feature_signature_expression([*context_features, "attack_cat", "label"])
        connection.execute(
            f"""
            CREATE TABLE frozen_feature_rows AS
            SELECT row_number() OVER () - 1 AS validation_row, *
            FROM read_parquet({sql_string(VALIDATION_FEATURES_PATH)})
            """
        )
        connection.execute(
            f"""
            CREATE TABLE frozen_keyed AS
            SELECT *,
                   row_number() OVER (
                       PARTITION BY signature ORDER BY validation_row
                   ) AS duplicate_rank
            FROM (
                SELECT *, {signature} AS signature
                FROM frozen_feature_rows
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE identity_keyed AS
            SELECT *,
                   row_number() OVER (
                       PARTITION BY signature
                       ORDER BY timestamp, flow_id, src_ip, dst_ip
                   ) AS duplicate_rank
            FROM (
                SELECT *, {signature} AS signature
                FROM enriched_identity
            )
            """
        )

        duplicate_stats = connection.execute(
            """
            WITH groups AS (
                SELECT signature, count(*) AS members
                FROM frozen_keyed GROUP BY signature
            )
            SELECT
                count(*) FILTER (WHERE members > 1),
                coalesce(sum(members) FILTER (WHERE members > 1), 0),
                max(members)
            FROM groups
            """
        ).fetchone()

        mismatch_terms = [
            f"(f.{quote_identifier(column)} IS DISTINCT FROM i.{quote_identifier(column)})"
            for column in [*context_features, "attack_cat", "label"]
        ]
        mismatch_expression = " OR ".join(mismatch_terms)
        alignment_stats = connection.execute(
            f"""
            SELECT
                count(*) AS matched_rows,
                count(*) FILTER (WHERE {mismatch_expression}) AS feature_mismatches,
                count(*) FILTER (
                    WHERE s.label IS DISTINCT FROM f.label
                       OR s.attack_cat IS DISTINCT FROM f.attack_cat
                ) AS score_target_mismatches,
                count(DISTINCT f.validation_row) AS distinct_feature_rows,
                count(DISTINCT i.flow_id || '|' || CAST(i.timestamp AS VARCHAR)) AS distinct_flow_time_keys
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(VALIDATION_SCORES_PATH)}) s
              ON s.validation_row = f.validation_row
            """
        ).fetchone()
        if alignment_stats is None:
            raise RuntimeError("Identity alignment returned no statistics")
        if tuple(map(int, alignment_stats[:4])) != (
            EXPECTED_VALIDATION_ROWS,
            0,
            0,
            EXPECTED_VALIDATION_ROWS,
        ):
            raise AssertionError(f"Identity alignment failed: {alignment_stats}")

        aligned = connection.execute(
            f"""
            SELECT
                f.validation_row,
                i.timestamp,
                i.flow_id,
                i.src_ip,
                i.dst_ip,
                i."Src Port"::BIGINT AS src_port,
                i."Dst Port"::BIGINT AS dst_port,
                i."Protocol"::BIGINT AS protocol,
                s.attack_cat,
                s.label::UTINYINT AS label,
                s.context_attack_score::DOUBLE AS attack_score,
                (s.context_attack_score >= {FROZEN_THRESHOLD})::UTINYINT AS predicted_class
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(VALIDATION_SCORES_PATH)}) s
              ON s.validation_row = f.validation_row
            ORDER BY f.validation_row
            """
        ).fetchdf()
    finally:
        connection.close()

    if len(aligned) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError("Aligned validation frame has the wrong size")
    if aligned["validation_row"].tolist() != list(range(EXPECTED_VALIDATION_ROWS)):
        raise AssertionError("Aligned validation rows are not in frozen score order")
    if aligned["timestamp"].dt.strftime("%Y-%m-%d").nunique() != 1:
        raise AssertionError("Aligned frame contains more than one capture date")
    return aligned, {
        "matched_rows": int(alignment_stats[0]),
        "feature_value_mismatches": int(alignment_stats[1]),
        "score_target_mismatches": int(alignment_stats[2]),
        "distinct_feature_rows": int(alignment_stats[3]),
        "distinct_flow_time_keys": int(alignment_stats[4]),
        "duplicate_feature_fingerprint_groups": int(duplicate_stats[0]),
        "rows_in_duplicate_feature_fingerprints": int(duplicate_stats[1]),
        "maximum_duplicate_fingerprint_size": int(duplicate_stats[2]),
        "signature_columns": len(context_features) + 2,
        "runtime_seconds": time.perf_counter() - started,
        "minimum_timestamp": routed_stats[1].isoformat(),
        "maximum_timestamp": routed_stats[2].isoformat(),
    }


def assign_incidents(
    flows: pd.DataFrame,
    key_columns: Iterable[str],
    window_seconds: int,
) -> pd.DataFrame:
    """Assign deterministic per-key temporal sessions to a flow subset."""

    keys = list(key_columns)
    if not keys:
        raise ValueError("At least one incident key is required")
    if window_seconds <= 0:
        raise ValueError("Incident window must be positive")
    required = {"timestamp", "validation_row", *keys}
    missing = sorted(required - set(flows.columns))
    if missing:
        raise ValueError(f"Incident input is missing columns: {missing}")
    if flows.empty:
        result = flows.copy()
        result["incident_id"] = pd.Series(dtype="object")
        return result

    ordered = flows.sort_values(
        [*keys, "timestamp", "validation_row"], kind="mergesort"
    ).copy()
    groupers = [ordered[column] for column in keys]
    gap_seconds = ordered.groupby(keys, sort=False, dropna=False)["timestamp"].diff()
    new_session = gap_seconds.isna() | gap_seconds.dt.total_seconds().gt(window_seconds)
    session_number = (
        new_session.groupby(groupers, sort=False, dropna=False).cumsum().astype(np.int64)
    )
    entity_number = ordered.groupby(keys, sort=False, dropna=False).ngroup()
    ordered["incident_id"] = (
        entity_number.astype(str) + ":" + session_number.astype(str)
    )
    return ordered


def incident_table(
    alert_flows: pd.DataFrame,
    key_columns: Iterable[str],
    window_seconds: int,
) -> pd.DataFrame:
    keys = list(key_columns)
    assigned = assign_incidents(alert_flows, keys, window_seconds)
    grouped = assigned.groupby("incident_id", sort=False, observed=True)
    incidents = grouped.agg(
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        flow_count=("validation_row", "size"),
        attack_flows=("label", "sum"),
        maximum_attack_score=("attack_score", "max"),
    ).reset_index()
    incidents["normal_flows"] = incidents["flow_count"] - incidents["attack_flows"]
    incidents["mixed"] = (incidents["attack_flows"] > 0) & (
        incidents["normal_flows"] > 0
    )
    incidents["false_positive"] = incidents["attack_flows"] == 0
    incidents["attack_only"] = (incidents["attack_flows"] > 0) & (
        incidents["normal_flows"] == 0
    )
    incidents["true_positive_including_mixed"] = incidents["attack_flows"] > 0

    key_values = grouped[keys].first().reset_index()
    incidents = incidents.merge(key_values, on="incident_id", how="left", validate="1:1")
    return incidents


def reference_attack_metrics(
    all_flows: pd.DataFrame,
    key_columns: Iterable[str],
    window_seconds: int,
    policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    attacks = all_flows.loc[all_flows["label"] == 1].copy()
    for category in ("ALL_ATTACKS", *ATTACK_CATEGORIES):
        category_flows = (
            attacks if category == "ALL_ATTACKS" else attacks.loc[attacks["attack_cat"] == category]
        )
        assigned = assign_incidents(category_flows, key_columns, window_seconds)
        grouped = assigned.groupby("incident_id", sort=False, observed=True)
        reference_incidents = grouped["predicted_class"].max()
        detected_incidents = int(reference_incidents.sum())
        total_incidents = int(len(reference_incidents))
        detected_flows = int(category_flows["predicted_class"].sum())
        total_flows = int(len(category_flows))
        rows.append(
            {
                "policy": policy,
                "policy_definition": POLICY_DESCRIPTIONS[policy],
                "window_seconds": window_seconds,
                "window_label": WINDOW_LABELS[window_seconds],
                "attack_category": category,
                "attack_flows": total_flows,
                "detected_attack_flows": detected_flows,
                "missed_attack_flows": total_flows - detected_flows,
                "attack_incidents": total_incidents,
                "detected_attack_incidents": detected_incidents,
                "missed_attack_incidents": total_incidents - detected_incidents,
                "flow_recall": detected_flows / total_flows if total_flows else np.nan,
                "incident_recall": (
                    detected_incidents / total_incidents if total_incidents else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def configuration_metrics(
    all_flows: pd.DataFrame,
    alert_flows: pd.DataFrame,
    policy: str,
    window_seconds: int,
    capture_hours: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    keys = POLICIES[policy]
    incidents = incident_table(alert_flows, keys, window_seconds)
    category_rows = reference_attack_metrics(
        all_flows, keys, window_seconds, policy
    )
    overall = category_rows.loc[category_rows["attack_category"] == "ALL_ATTACKS"].iloc[0]

    total_alerts = len(alert_flows)
    normal_alerts = int((alert_flows["label"] == 0).sum())
    attack_alerts = total_alerts - normal_alerts
    total_incidents = len(incidents)
    false_positive_incidents = int(incidents["false_positive"].sum())
    mixed_incidents = int(incidents["mixed"].sum())
    attack_only_incidents = int(incidents["attack_only"].sum())
    true_positive_incidents = int(incidents["true_positive_including_mixed"].sum())
    normal_flows_in_mixed = int(
        incidents.loc[incidents["mixed"], "normal_flows"].sum()
    )
    attack_flows_in_mixed = int(
        incidents.loc[incidents["mixed"], "attack_flows"].sum()
    )

    normal_only_sessions = len(
        assign_incidents(
            alert_flows.loc[alert_flows["label"] == 0], keys, window_seconds
        )["incident_id"].unique()
    )
    attack_only_sessions = len(
        assign_incidents(
            alert_flows.loc[alert_flows["label"] == 1], keys, window_seconds
        )["incident_id"].unique()
    )
    sizes = incidents["flow_count"].to_numpy(dtype=float)
    row = {
        "policy": policy,
        "policy_definition": POLICY_DESCRIPTIONS[policy],
        "grouping_key_count": len(keys),
        "window_seconds": window_seconds,
        "window_label": WINDOW_LABELS[window_seconds],
        "total_validation_flows": len(all_flows),
        "total_alert_flows": total_alerts,
        "normal_alert_flows": normal_alerts,
        "detected_attack_flows": attack_alerts,
        "total_incidents": total_incidents,
        "true_positive_incidents": true_positive_incidents,
        "attack_only_incidents": attack_only_incidents,
        "false_positive_incidents": false_positive_incidents,
        "mixed_incidents": mixed_incidents,
        "normal_alert_flows_in_mixed_incidents": normal_flows_in_mixed,
        "attack_alert_flows_in_mixed_incidents": attack_flows_in_mixed,
        "normal_alert_absorption_into_mixed_fraction": safe_ratio(
            normal_flows_in_mixed, normal_alerts
        ),
        "average_flows_per_incident": float(sizes.mean()),
        "median_flows_per_incident": float(np.median(sizes)),
        "p95_flows_per_incident": float(np.quantile(sizes, 0.95)),
        "maximum_flows_per_incident": int(sizes.max()),
        "flow_level_fp_count": normal_alerts,
        "incident_level_fp_count": false_positive_incidents,
        "fp_compression_ratio": safe_ratio(normal_alerts, false_positive_incidents),
        "alert_reduction_percentage": 1.0 - total_incidents / total_alerts,
        "normal_alert_incidents_when_grouped_separately": normal_only_sessions,
        "detected_attack_incidents_when_grouped_separately": attack_only_sessions,
        "normal_flow_compression": safe_ratio(normal_alerts, normal_only_sessions),
        "attack_flow_compression": safe_ratio(attack_alerts, attack_only_sessions),
        "reference_attack_incidents": int(overall["attack_incidents"]),
        "detected_reference_attack_incidents": int(overall["detected_attack_incidents"]),
        "missed_reference_attack_incidents": int(overall["missed_attack_incidents"]),
        "overall_attack_flow_recall": float(overall["flow_recall"]),
        "overall_attack_incident_recall": float(overall["incident_recall"]),
        "capture_hours": capture_hours,
        "flows_per_hour_before_aggregation": len(all_flows) / capture_hours,
        "alerts_per_hour_before_aggregation": total_alerts / capture_hours,
        "incidents_per_hour_after_aggregation": total_incidents / capture_hours,
        "false_positive_incidents_per_hour": false_positive_incidents / capture_hours,
        "true_or_mixed_security_incidents_per_hour": true_positive_incidents / capture_hours,
        "operational_candidate": False,
        "candidate_rank": np.nan,
        "recommended": False,
    }
    if total_incidents != attack_only_incidents + false_positive_incidents + mixed_incidents:
        raise AssertionError("Exclusive incident labels do not partition all incidents")
    if true_positive_incidents != attack_only_incidents + mixed_incidents:
        raise AssertionError("Inclusive true-positive incident count is inconsistent")
    return row, category_rows, incidents


def select_operational_candidates(results: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select one semantics-aware candidate per policy.

    Policy A has no service/port discriminator, so windows beyond 60 seconds
    carry a much higher risk of absorbing unrelated normal activity into mixed
    attack incidents.  Service-aware B/C are allowed up to five minutes.  The
    15-minute variants remain in the report but never enter the shortlist.
    """

    maximum_candidate_window = {"A": 60, "B": 300, "C": 300}
    per_policy: list[pd.Series] = []
    for policy in POLICIES:
        subset = results.loc[
            (results["policy"] == policy)
            & (results["window_seconds"] <= maximum_candidate_window[policy])
            & (results["overall_attack_incident_recall"] >= NEAR_PERFECT_INCIDENT_RECALL)
        ]
        if subset.empty:
            subset = results.loc[
                (results["policy"] == policy)
                & (results["window_seconds"] <= maximum_candidate_window[policy])
            ].sort_values(
                ["overall_attack_incident_recall", "false_positive_incidents"],
                ascending=[False, True],
            )
        selected = subset.sort_values(
            ["false_positive_incidents", "window_seconds"], ascending=[True, True]
        ).iloc[0]
        per_policy.append(selected)

    candidates = pd.DataFrame(per_policy)
    service_aware = candidates.loc[candidates["policy"].isin(["B", "C"])]
    best_service_fp = int(service_aware["false_positive_incidents"].min())
    best_service_recall = float(service_aware["overall_attack_incident_recall"].max())
    comparable_service = service_aware.loc[
        (service_aware["false_positive_incidents"] <= best_service_fp * 1.10)
        & (
            service_aware["overall_attack_incident_recall"]
            >= best_service_recall - 0.001
        )
    ]
    # B is the simpler of the comparable service-aware keys.  A stays a useful
    # aggressive candidate, but is not the default because it merges ports.
    recommended = comparable_service.sort_values(
        ["grouping_key_count", "false_positive_incidents"], ascending=[True, True]
    ).iloc[0]
    rank_order = {
        str(recommended["policy"]): 1,
        "C" if recommended["policy"] == "B" else "B": 2,
        "A": 3,
    }
    candidates["candidate_rank"] = candidates["policy"].map(rank_order).astype(int)
    candidates = candidates.sort_values("candidate_rank")

    updated = results.copy()
    summaries: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        mask = (updated["policy"] == candidate["policy"]) & (
            updated["window_seconds"] == candidate["window_seconds"]
        )
        updated.loc[mask, "operational_candidate"] = True
        updated.loc[mask, "candidate_rank"] = int(candidate["candidate_rank"])
        is_recommended = bool(
            candidate["policy"] == recommended["policy"]
            and candidate["window_seconds"] == recommended["window_seconds"]
        )
        updated.loc[mask, "recommended"] = is_recommended
        summaries.append(
            {
                "rank": int(candidate["candidate_rank"]),
                "policy": str(candidate["policy"]),
                "policy_definition": str(candidate["policy_definition"]),
                "window_seconds": int(candidate["window_seconds"]),
                "window_label": str(candidate["window_label"]),
                "false_positive_incidents": int(candidate["false_positive_incidents"]),
                "fp_compression_ratio": float(candidate["fp_compression_ratio"]),
                "alert_reduction_percentage": float(candidate["alert_reduction_percentage"]),
                "overall_attack_incident_recall": float(
                    candidate["overall_attack_incident_recall"]
                ),
                "normal_alert_absorption_into_mixed_fraction": float(
                    candidate["normal_alert_absorption_into_mixed_fraction"]
                ),
                "recommended": is_recommended,
            }
        )
    return updated, summaries


def top_fp_patterns(
    incidents_by_configuration: dict[tuple[str, int], pd.DataFrame],
    candidates: list[dict[str, Any]],
    flow_level_fp_count: int,
    false_positive_flows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    dominance: list[dict[str, Any]] = []
    for candidate in candidates:
        policy = candidate["policy"]
        window = candidate["window_seconds"]
        keys = list(POLICIES[policy])
        fp_incidents = incidents_by_configuration[(policy, window)].loc[
            lambda frame: frame["false_positive"]
        ].copy()
        grouped = (
            fp_incidents.groupby(keys, dropna=False, observed=True)
            .agg(fp_flows=("flow_count", "sum"), fp_incidents=("incident_id", "size"))
            .reset_index()
            .sort_values(["fp_flows", "fp_incidents"], ascending=[False, False])
        )
        grouped["rank"] = np.arange(1, len(grouped) + 1)
        grouped["cumulative_fp_flows"] = grouped["fp_flows"].cumsum()
        grouped["cumulative_percentage_all_fp_flows"] = (
            grouped["cumulative_fp_flows"] / flow_level_fp_count
        )
        grouped["policy"] = policy
        grouped["policy_definition"] = POLICY_DESCRIPTIONS[policy]
        grouped["window_seconds"] = window
        grouped["window_label"] = WINDOW_LABELS[window]
        for column in ("src_ip", "dst_ip", "dst_port", "protocol"):
            if column not in grouped:
                grouped[column] = pd.NA
        if policy == "A":
            grouped["service_pattern"] = "all protocols/ports"
        elif policy == "B":
            grouped["service_pattern"] = (
                "*/" + grouped["dst_port"].astype("Int64").astype(str)
            )
        else:
            grouped["service_pattern"] = (
                grouped["protocol"].astype("Int64").astype(str)
                + "/"
                + grouped["dst_port"].astype("Int64").astype(str)
            )
        top = grouped.head(20).copy()
        frames.append(top)

        total_fp_incident_flows = int(grouped["fp_flows"].sum())
        top20_flows = int(top["fp_flows"].sum())
        dominance.append(
            {
                "policy": policy,
                "window_seconds": window,
                "distinct_fp_keys": int(len(grouped)),
                "fp_flows_in_false_positive_incidents": total_fp_incident_flows,
                "top_20_fp_flows": top20_flows,
                "top_20_share_of_all_flow_level_fps": top20_flows / flow_level_fp_count,
                "top_20_share_of_fp_only_incident_flows": safe_ratio(
                    top20_flows, total_fp_incident_flows
                ),
                "small_number_of_patterns_dominates": bool(
                    safe_ratio(top20_flows, total_fp_incident_flows) >= 0.50
                ),
            }
        )

    reference_keys = ["src_ip", "dst_ip", "dst_port", "protocol"]
    reference = (
        false_positive_flows.groupby(reference_keys, dropna=False, observed=True)
        .size()
        .rename("fp_flows")
        .reset_index()
        .sort_values("fp_flows", ascending=False)
    )
    reference["fp_incidents"] = pd.NA
    reference["rank"] = np.arange(1, len(reference) + 1)
    reference["cumulative_fp_flows"] = reference["fp_flows"].cumsum()
    reference["cumulative_percentage_all_fp_flows"] = (
        reference["cumulative_fp_flows"] / flow_level_fp_count
    )
    reference["policy"] = "FLOW_LEVEL_REFERENCE"
    reference["policy_definition"] = "src_ip + dst_ip + dst_port + protocol"
    reference["window_seconds"] = 0
    reference["window_label"] = "no aggregation"
    reference["service_pattern"] = (
        reference["protocol"].astype("Int64").astype(str)
        + "/"
        + reference["dst_port"].astype("Int64").astype(str)
    )
    reference_top = reference.head(20).copy()
    frames.append(reference_top)
    reference_top20_flows = int(reference_top["fp_flows"].sum())
    dominance.append(
        {
            "policy": "FLOW_LEVEL_REFERENCE",
            "window_seconds": 0,
            "distinct_fp_keys": int(len(reference)),
            "fp_flows_in_false_positive_incidents": flow_level_fp_count,
            "top_20_fp_flows": reference_top20_flows,
            "top_20_share_of_all_flow_level_fps": (
                reference_top20_flows / flow_level_fp_count
            ),
            "top_20_share_of_fp_only_incident_flows": (
                reference_top20_flows / flow_level_fp_count
            ),
            "small_number_of_patterns_dominates": bool(
                reference_top20_flows / flow_level_fp_count >= 0.50
            ),
        }
    )
    columns = [
        "policy",
        "policy_definition",
        "window_seconds",
        "window_label",
        "rank",
        "src_ip",
        "dst_ip",
        "dst_port",
        "protocol",
        "service_pattern",
        "fp_flows",
        "fp_incidents",
        "cumulative_fp_flows",
        "cumulative_percentage_all_fp_flows",
    ]
    return pd.concat(frames, ignore_index=True)[columns], dominance


def validate_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], dict[str, str]]:
    paths = (
        RAW_PATH,
        VALIDATION_FEATURES_PATH,
        FEATURE_MANIFEST_PATH,
        VALIDATION_SCORES_PATH,
        ABLATION_METRICS_PATH,
        CONTEXT_MODEL_PATH,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    contract = feature_manifest["feature_contract"]
    baseline = list(contract["baseline_v2_features"])
    context = [
        *baseline,
        *contract["context_v2_static_behavioral_features"],
        *contract["context_v2_temporal_features"],
    ]
    if len(baseline) != 76 or len(context) != 128:
        raise AssertionError("Frozen feature contract changed")
    if ablation["scope"]["locked_holdout_loaded"]:
        raise AssertionError("Frozen ablation manifest reports forbidden holdout access")
    selected = ablation["models"]["context_v2"]["selected_operating_point"]
    if not np.isclose(float(selected["threshold"]), FROZEN_THRESHOLD):
        raise AssertionError("Frozen Context threshold changed")

    hashes = {
        "model": sha256_file(CONTEXT_MODEL_PATH),
        "scores": sha256_file(VALIDATION_SCORES_PATH),
        "validation_features": sha256_file(VALIDATION_FEATURES_PATH),
    }
    expected_model_hash = ablation["models"]["context_v2"]["continuous_metrics"][
        "model_sha256"
    ]
    expected_score_hash = ablation["artifacts"]["validation_scores_sha256"]
    expected_validation_hash = feature_manifest["outputs"]["validation"]["sha256"]
    if hashes != {
        "model": expected_model_hash,
        "scores": expected_score_hash,
        "validation_features": expected_validation_hash,
    }:
        raise AssertionError("One or more frozen input hashes changed")
    return feature_manifest, ablation, baseline, context, hashes


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    total_started = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    feature_manifest, ablation, baseline, context, hashes_before = validate_frozen_inputs()
    print("Routing February 17 identity rows from the combined raw CSV...", flush=True)
    routing = route_validation_rows(RAW_PATH, ROUTED_VALIDATION_CSV)
    print("Rebuilding tie-safe February 17 context for identity alignment...", flush=True)
    flows, alignment = restore_validation_identities(
        ROUTED_VALIDATION_CSV,
        context,
        baseline,
        args.memory_limit,
        args.threads,
    )

    score_prediction = (flows["attack_score"] >= FROZEN_THRESHOLD).astype(np.uint8)
    if not np.array_equal(score_prediction, flows["predicted_class"].to_numpy()):
        raise AssertionError("Frozen threshold predictions are inconsistent")
    if set(flows["label"].unique()) != {0, 1}:
        raise AssertionError("Validation target is not binary")
    alerts = flows.loc[flows["predicted_class"] == 1].copy()
    flow_level = {
        "validation_flows": len(flows),
        "normal_flows": int((flows["label"] == 0).sum()),
        "attack_flows": int((flows["label"] == 1).sum()),
        "threshold": FROZEN_THRESHOLD,
        "alert_flows": len(alerts),
        "true_positive_flows": int((alerts["label"] == 1).sum()),
        "false_positive_flows": int((alerts["label"] == 0).sum()),
        "false_negative_flows": int(
            ((flows["label"] == 1) & (flows["predicted_class"] == 0)).sum()
        ),
    }
    flow_level["recall"] = flow_level["true_positive_flows"] / flow_level["attack_flows"]
    flow_level["fpr"] = flow_level["false_positive_flows"] / flow_level["normal_flows"]
    expected = ablation["models"]["context_v2"]["selected_operating_point"]
    for name, expected_name in (
        ("false_positive_flows", "false_positives"),
        ("false_negative_flows", "false_negatives"),
    ):
        if flow_level[name] != int(expected[expected_name]):
            raise AssertionError(f"Frozen flow metric changed: {name}")
    if not np.isclose(flow_level["recall"], float(expected["recall"])):
        raise AssertionError("Frozen flow recall changed")
    if not np.isclose(flow_level["fpr"], float(expected["fpr"])):
        raise AssertionError("Frozen flow FPR changed")
    flow_level["pr_auc"] = float(
        ablation["models"]["context_v2"]["continuous_metrics"][
            "pr_auc_average_precision"
        ]
    )

    minimum_timestamp = flows["timestamp"].min()
    maximum_timestamp = flows["timestamp"].max()
    capture_seconds = (maximum_timestamp - minimum_timestamp).total_seconds() + 1.0
    capture_hours = capture_seconds / 3600.0
    if capture_hours <= 0:
        raise AssertionError("Invalid validation capture duration")

    print("Evaluating 12 deterministic policy/window configurations...", flush=True)
    result_rows: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    incident_tables: dict[tuple[str, int], pd.DataFrame] = {}
    for policy in POLICIES:
        for window_seconds in WINDOWS_SECONDS:
            row, category_rows, incidents = configuration_metrics(
                flows, alerts, policy, window_seconds, capture_hours
            )
            result_rows.append(row)
            category_frames.append(category_rows)
            incident_tables[(policy, window_seconds)] = incidents
            print(
                f"  {policy}/{WINDOW_LABELS[window_seconds]}: "
                f"incidents={row['total_incidents']:,}, "
                f"FP incidents={row['false_positive_incidents']:,}, "
                f"attack incident recall={row['overall_attack_incident_recall']:.6f}",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    results, candidates = select_operational_candidates(results)
    category_metrics = pd.concat(category_frames, ignore_index=True)
    fp_patterns, pattern_dominance = top_fp_patterns(
        incident_tables,
        candidates,
        flow_level["false_positive_flows"],
        alerts.loc[alerts["label"] == 0],
    )

    results = results.sort_values(["policy", "window_seconds"]).reset_index(drop=True)
    category_metrics = category_metrics.sort_values(
        ["policy", "window_seconds", "attack_category"]
    ).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)
    category_metrics.to_csv(CATEGORY_METRICS_PATH, index=False)
    fp_patterns.to_csv(FP_PATTERNS_PATH, index=False)

    hashes_after = {
        "model": sha256_file(CONTEXT_MODEL_PATH),
        "scores": sha256_file(VALIDATION_SCORES_PATH),
        "validation_features": sha256_file(VALIDATION_FEATURES_PATH),
    }
    if hashes_after != hashes_before:
        raise AssertionError("A frozen artifact changed during aggregation analysis")

    recommended = next(candidate for candidate in candidates if candidate["recommended"])
    manifest = {
        "stage": "v2 frozen Context incident-level aggregation analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "capture_date": CAPTURE_DATE,
            "classifier_retrained": False,
            "classifier_loaded": False,
            "threshold_changed": False,
            "features_changed": False,
            "locked_holdout_loaded": False,
            "locked_holdout_materialized": False,
            "locked_holdout_scored": False,
            "locked_holdout_evaluated": False,
        },
        "frozen_inputs": {
            "model_path": str(CONTEXT_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "frozen_model_sha256": hashes_before["model"],
            "score_path": str(VALIDATION_SCORES_PATH.relative_to(PROJECT_ROOT)),
            "frozen_score_artifact_sha256": hashes_before["scores"],
            "feb17_source_dataset_path": str(
                VALIDATION_FEATURES_PATH.relative_to(PROJECT_ROOT)
            ),
            "feb17_source_dataset_sha256": hashes_before["validation_features"],
            "combined_raw_source_sha256_from_frozen_feature_manifest": feature_manifest[
                "source"
            ]["sha256"],
            "routed_feb17_raw_rows_sha256": routing["sha256"],
            "threshold": FROZEN_THRESHOLD,
            "hashes_unchanged_after_analysis": hashes_after == hashes_before,
        },
        "identity_routing": {
            **routing,
            "combined_csv_is_not_chronologically_sorted": True,
            "rule": (
                "Inspect only raw Timestamp for routing; materialize and parse only "
                "rows whose timestamp date is 2015-02-17. Nonmatching rows are "
                "discarded as opaque bytes and never enter DuckDB or the analysis."
            ),
            "raw_identifiers_used_as_model_features": False,
        },
        "identity_alignment": alignment,
        "flow_level_baseline": flow_level,
        "grouping_definitions": {
            policy: {
                "columns": list(columns),
                "description": POLICY_DESCRIPTIONS[policy],
            }
            for policy, columns in POLICIES.items()
        },
        "evaluated_time_windows_seconds": list(WINDOWS_SECONDS),
        "timestamp_sorting_rules": {
            "primary": "Timestamp ascending within each grouping key",
            "deterministic_tie_breaker": "frozen validation_row ascending",
            "same_second_policy": (
                "Same-key alerts at the same timestamp have zero gap and remain in "
                "the same incident; no ordering inside a second is inferred."
            ),
            "session_boundary": (
                "A new incident starts for a key when the gap from its immediately "
                "preceding alert is greater than the configured window."
            ),
            "interleaved_other_keys": (
                "Other entity keys do not break a key's session; gaps are measured "
                "between consecutive alerts for that key."
            ),
        },
        "incident_labeling_rules": {
            "true_positive_incident": (
                "Contains at least one attack flow; this inclusive count includes "
                "mixed incidents."
            ),
            "false_positive_incident": "All member alert flows are normal.",
            "mixed_incident": "Contains both attack and normal alert flows.",
            "attack_only_incident": "All member alert flows are attacks.",
            "operational_fp_rule": (
                "Mixed incidents are not counted as false-positive incidents because "
                "at least one real attack justifies analyst attention."
            ),
        },
        "reference_attack_incident_rule": (
            "For each category (and overall), group all actual attack flows with the "
            "same policy/window using timestamps only; a reference incident is "
            "detected when at least one member flow crossed the frozen threshold."
        ),
        "compression_definitions": {
            "fp_compression_ratio": "flow-level FP / all-normal alert incidents",
            "normal_flow_compression": (
                "normal alert flows / sessions obtained by grouping normal alert "
                "flows separately"
            ),
            "attack_flow_compression": (
                "detected attack flows / sessions obtained by grouping detected "
                "attack flows separately"
            ),
        },
        "workload_denominator": {
            "minimum_timestamp": minimum_timestamp.isoformat(),
            "maximum_timestamp": maximum_timestamp.isoformat(),
            "capture_seconds_inclusive": capture_seconds,
            "capture_hours_inclusive": capture_hours,
        },
        "candidate_selection": {
            "near_perfect_incident_recall_gate": NEAR_PERFECT_INCIDENT_RECALL,
            "maximum_shortlist_window_by_policy_seconds": {
                "A": 60,
                "B": 300,
                "C": 300,
            },
            "merge_risk_control": (
                "Policy A omits destination port/protocol, so only <=60-second "
                "variants are shortlisted. Policy B/C retain a service proxy and "
                "may use <=5 minutes. All 15-minute results are reported but not "
                "shortlisted."
            ),
            "method": (
                "Choose the lowest-FP configuration per policy after the incident-"
                "recall and policy-specific merge-risk gates. Compare service-aware "
                "B/C candidates within 10% FP count and 0.1 percentage point recall; "
                "prefer the simpler grouping key. Keep A as an aggressive diagnostic "
                "candidate, not the default."
            ),
            "best_three": candidates,
            "recommended": recommended,
        },
        "fp_pattern_dominance": pattern_dominance,
        "service_field_note": (
            "CICFlowMeter_out.csv has no service-name column; Dst Port + Protocol is "
            "reported as the service-pattern proxy."
        ),
        "runtime": {
            "total_seconds": time.perf_counter() - total_started,
            "routing_seconds": routing["runtime_seconds"],
            "identity_alignment_seconds": alignment["runtime_seconds"],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "results": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "results_sha256": sha256_file(RESULTS_PATH),
            "category_metrics": str(CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "category_metrics_sha256": sha256_file(CATEGORY_METRICS_PATH),
            "fp_patterns": str(FP_PATTERNS_PATH.relative_to(PROJECT_ROOT)),
            "fp_patterns_sha256": sha256_file(FP_PATTERNS_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.keep_cache:
        ROUTED_VALIDATION_CSV.unlink(missing_ok=True)
        try:
            shutil.rmtree(CACHE_DIR)
        except FileNotFoundError:
            pass
        except OSError:
            # DuckDB/OneDrive can briefly retain directory metadata after the
            # database connection closes.  The routed CSV is already removed;
            # leftover spill metadata is non-authoritative and safe to retain.
            pass

    print("\nFrozen flow-level baseline:")
    print(json.dumps(flow_level, indent=2))
    print("\nOperational candidates:")
    print(pd.DataFrame(candidates).to_string(index=False))
    print(f"\nRecommended: Policy {recommended['policy']}, {recommended['window_label']}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Category metrics: {CATEGORY_METRICS_PATH}")
    print(f"FP patterns: {FP_PATTERNS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
