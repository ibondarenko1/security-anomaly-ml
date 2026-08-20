"""Verify raw, label-free Feb 17 flows through the complete product path.

This check is intentionally separate from ``verify_frozen_score_parity.py``.
That tool begins with frozen research features; this tool begins with a
Feb-17-only CICFlowMeter CSV, removes evaluation fields, rebuilds all 128
features through ``CausalTemporalFeatureBuilder``, and then scores them through
``FrozenModelBundle``.

The tool never routes a combined multi-day file. Its input must already contain
only February 17, 2015 rows, which prevents accidental Feb 18 access.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_temporal_features import (  # noqa: E402
    context_expressions as research_context_expressions,
)
from src.build_temporal_features import window_clause as research_window_clause  # noqa: E402
from src.security_anomaly.contracts import FeatureContract, sha256_file  # noqa: E402
from src.security_anomaly.model_bundle import FrozenModelBundle  # noqa: E402
from src.security_anomaly.temporal import CausalTemporalFeatureBuilder  # noqa: E402


DEFAULT_RAW_FEB17 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cic_unsw_nb15_v2"
    / "_product_parity_cache"
    / "feb17_raw.csv"
)
DEFAULT_FROZEN_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cic_unsw_nb15_v2"
    / "validation_features.parquet"
)
DEFAULT_REFERENCE_SCORES = PROJECT_ROOT / "models" / "v2_validation_scores.parquet"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "v2_context_random_forest.joblib"
DEFAULT_MANIFEST = PROJECT_ROOT / "contracts" / "model-manifest-context-rf-v2.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
DEFAULT_TEMP_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cic_unsw_nb15_v2"
    / "_product_parity_cache"
    / "duckdb_temp"
)

CAPTURE_DATE = "2015-02-17"
EXPECTED_ROWS = 498_890
EXPECTED_RAW_FEB17_SHA256 = (
    "d1a016231c1a3ffa554b6faa8459f12a7743ae320a2a2b4a070a9da8bc56e54d"
)
EXPECTED_FROZEN_FEATURES_SHA256 = (
    "bb33d2d5379f801d87f4806e8543641bd5dffeaaaa37d045ce33d401c05aec0a"
)
EXPECTED_REFERENCE_SCORES_SHA256 = (
    "e471dc6953851819345d9850e88e13d81aaaeaba701a5bb815c038b741e3649b"
)
FLOW_THRESHOLD = 0.10
DEFAULT_FEATURE_ATOL = 0.0
PARITY_SOURCE_ROW = "_parity_source_row_id"
EVALUATION_ONLY_COLUMNS = frozenset(
    {
        "Label",
        "label",
        "attack_cat",
        "attack_score",
        "baseline_attack_score",
        "context_attack_score",
        "predicted_class",
        "prediction",
    }
)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-feb17",
        type=Path,
        default=DEFAULT_RAW_FEB17,
        help=(
            "Feb-17-only raw CICFlowMeter CSV. A combined multi-day source is "
            "rejected by date validation and must not be supplied."
        ),
    )
    parser.add_argument(
        "--frozen-features", type=Path, default=DEFAULT_FROZEN_FEATURES
    )
    parser.add_argument(
        "--reference-scores", type=Path, default=DEFAULT_REFERENCE_SCORES
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--threads", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument(
        "--feature-atol",
        type=float,
        default=DEFAULT_FEATURE_ATOL,
        help=(
            "Absolute tolerance for numeric feature comparison after conversion "
            "to the contract's float32 model-matrix dtype (default: exact)."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional local JSON output. Do not commit dataset-derived reports.",
    )
    return parser.parse_args()


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    memory_limit: str,
    threads: int,
    temp_directory: Path,
) -> None:
    if threads <= 0:
        raise ValueError("--threads must be positive")
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_string(temp_directory)}")


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )


def load_feb17_only_csv(path: Path) -> pd.DataFrame:
    """Load one already-isolated day without opening any multi-day source."""

    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            SELECT *
            FROM read_csv_auto(
                {sql_string(path)},
                header = true,
                sample_size = -1,
                parallel = false
            )
            """
        ).fetchdf()
    finally:
        connection.close()
    if len(frame) != EXPECTED_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROWS:,} Feb 17 rows, loaded {len(frame):,}"
        )
    timestamps = pd.to_datetime(
        frame["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="raise"
    )
    dates = timestamps.dt.strftime("%Y-%m-%d").unique().tolist()
    if dates != [CAPTURE_DATE]:
        raise AssertionError(
            "Raw parity input must contain only 2015-02-17 rows; "
            f"observed dates: {dates}"
        )
    return frame


def remove_evaluation_columns(
    frame: pd.DataFrame, contract: FeatureContract
) -> tuple[pd.DataFrame, list[str]]:
    """Remove ground truth in place and reject every unrelated extra field."""

    declared = set(EVALUATION_ONLY_COLUMNS)
    declared.update(contract.document["input"]["ignored_evaluation_columns"])
    removed = [column for column in frame.columns if column in declared]
    for column in removed:
        frame.pop(column)

    allowed = set(contract.required_input_columns)
    allowed.update(contract.document["input"]["optional_preserved_columns"])
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError(
            "Feb 17 product input contains undeclared non-user fields after "
            f"ground-truth removal: {unexpected}"
        )
    remaining_ground_truth = sorted(set(frame.columns) & declared)
    if remaining_ground_truth:
        raise AssertionError(
            f"Evaluation fields remain in product input: {remaining_ground_truth}"
        )
    missing = [name for name in contract.required_input_columns if name not in frame]
    if missing:
        raise ValueError(f"Raw product input is missing required fields: {missing}")
    return frame, sorted(removed)


def research_enriched_query(contract: FeatureContract) -> str:
    """Return the frozen research SQL while preserving a stable source row."""

    baseline_sql = ",\n        ".join(
        quote_identifier(name) for name in contract.baseline_features
    )
    context_sql = ",\n        ".join(research_context_expressions())
    selected_sql = ",\n    ".join(
        quote_identifier(name) for name in contract.model_features
    )
    return f"""
    WITH prepared AS (
        SELECT
            *,
            strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
            ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE
                AS flow_bytes,
            ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
        FROM alignment_input
    ),
    enriched AS (
        SELECT
            {PARITY_SOURCE_ROW},
            event_ts,
            {baseline_sql},
            "Src Port",
            "Dst Port",
            "Protocol",
            CASE WHEN "Src Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT
                AS src_port_well_known,
            CASE WHEN "Src Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT
                AS src_port_registered,
            CASE WHEN "Src Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT
                AS src_port_ephemeral,
            CASE WHEN "Dst Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT
                AS dst_port_well_known,
            CASE WHEN "Dst Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT
                AS dst_port_registered,
            CASE WHEN "Dst Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT
                AS dst_port_ephemeral,
            {context_sql}
        FROM prepared
        {research_window_clause()}
    )
    SELECT {PARITY_SOURCE_ROW}, {selected_sql}
    FROM enriched
    """


def fingerprint_expression(features: Sequence[str], alias: str = "") -> str:
    """Create a canonical SHA-256 over the complete ordered numeric row."""

    prefix = f"{alias}." if alias else ""
    fields = ", ".join(
        f"f{index:03d} := {prefix}{quote_identifier(name)}"
        for index, name in enumerate(features)
    )
    return f"sha256(to_json(struct_pack({fields})))"


def build_lossless_alignment(
    connection: duckdb.DuckDBPyConnection,
    *,
    frozen_features_path: Path,
    contract: FeatureContract,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map stable raw source rows to frozen physical Parquet rows.

    The mapping is independent of the product builder. It regenerates the
    frozen research feature path, fingerprints the complete ordered 128-value
    row with SHA-256, and verifies every joined value directly. Identical
    duplicate rows are ranked deterministically by source row and frozen row;
    exchanging them cannot change features or model scores.
    """

    features = contract.model_features
    connection.execute(
        f"CREATE TEMP TABLE research_generated AS {research_enriched_query(contract)}"
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE frozen_rows AS
        SELECT file_row_number::BIGINT AS validation_row, * EXCLUDE (file_row_number)
        FROM read_parquet(
            {sql_string(frozen_features_path)},
            file_row_number = true
        )
        """
    )
    research_fingerprint = fingerprint_expression(features)
    frozen_fingerprint = fingerprint_expression(features)
    connection.execute(
        f"""
        CREATE TEMP TABLE research_keyed AS
        SELECT *,
               row_number() OVER (
                   PARTITION BY row_fingerprint ORDER BY {PARITY_SOURCE_ROW}
               ) AS duplicate_rank
        FROM (
            SELECT *, {research_fingerprint} AS row_fingerprint
            FROM research_generated
        )
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE frozen_keyed AS
        SELECT *,
               row_number() OVER (
                   PARTITION BY row_fingerprint ORDER BY validation_row
               ) AS duplicate_rank
        FROM (
            SELECT *, {frozen_fingerprint} AS row_fingerprint
            FROM frozen_rows
        )
        """
    )

    duplicate_stats = connection.execute(
        """
        WITH groups AS (
            SELECT row_fingerprint, count(*) AS members
            FROM frozen_keyed
            GROUP BY row_fingerprint
        )
        SELECT
            count(*) FILTER (WHERE members > 1),
            coalesce(sum(members) FILTER (WHERE members > 1), 0),
            coalesce(max(members), 0)
        FROM groups
        """
    ).fetchone()
    unmatched_research = int(
        connection.execute(
            """
            SELECT count(*)
            FROM research_keyed r
            LEFT JOIN frozen_keyed f USING (row_fingerprint, duplicate_rank)
            WHERE f.validation_row IS NULL
            """
        ).fetchone()[0]
    )
    unmatched_frozen = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM frozen_keyed f
            LEFT JOIN research_keyed r USING (row_fingerprint, duplicate_rank)
            WHERE r.{PARITY_SOURCE_ROW} IS NULL
            """
        ).fetchone()[0]
    )
    mismatch_expression = " OR ".join(
        f"r.{quote_identifier(name)} IS DISTINCT FROM f.{quote_identifier(name)}"
        for name in features
    )
    matched_stats = connection.execute(
        f"""
        SELECT
            count(*) AS matched_rows,
            count(DISTINCT r.{PARITY_SOURCE_ROW}) AS distinct_source_rows,
            count(DISTINCT f.validation_row) AS distinct_frozen_rows,
            count(*) FILTER (WHERE {mismatch_expression}) AS direct_value_mismatches
        FROM research_keyed r
        JOIN frozen_keyed f USING (row_fingerprint, duplicate_rank)
        """
    ).fetchone()
    if matched_stats is None:
        raise RuntimeError("Feature alignment returned no statistics")
    expected = (expected_rows, expected_rows, expected_rows, 0)
    observed = tuple(map(int, matched_stats))
    if observed != expected or unmatched_research or unmatched_frozen:
        raise AssertionError(
            "Lossless source-row alignment failed: "
            f"matched={observed}, unmatched_research={unmatched_research}, "
            f"unmatched_frozen={unmatched_frozen}"
        )

    mapping = connection.execute(
        f"""
        SELECT f.validation_row, r.{PARITY_SOURCE_ROW}
        FROM research_keyed r
        JOIN frozen_keyed f USING (row_fingerprint, duplicate_rank)
        ORDER BY f.validation_row
        """
    ).fetchnumpy()
    validation_rows = np.asarray(mapping["validation_row"], dtype=np.int64)
    source_rows = np.asarray(mapping[PARITY_SOURCE_ROW], dtype=np.int64)
    if not np.array_equal(validation_rows, np.arange(expected_rows, dtype=np.int64)):
        raise AssertionError("Frozen validation rows are not a complete zero-based range")
    if not np.array_equal(np.sort(source_rows), np.arange(expected_rows, dtype=np.int64)):
        raise AssertionError("Source-row mapping is not a lossless zero-based permutation")
    return source_rows, {
        "method": (
            "stable raw source row -> independently regenerated frozen research "
            "features -> SHA-256 of all ordered 128 values -> direct 128-value "
            "verification -> frozen physical Parquet row"
        ),
        "matched_rows": int(matched_stats[0]),
        "distinct_source_rows": int(matched_stats[1]),
        "distinct_frozen_rows": int(matched_stats[2]),
        "direct_value_mismatches": int(matched_stats[3]),
        "unmatched_research_rows": unmatched_research,
        "unmatched_frozen_rows": unmatched_frozen,
        "duplicate_identical_feature_groups": int(duplicate_stats[0]),
        "rows_in_duplicate_identical_feature_groups": int(duplicate_stats[1]),
        "maximum_identical_duplicate_group_size": int(duplicate_stats[2]),
        "duplicate_policy": (
            "Identical complete 128-feature rows are ranked by source row and "
            "frozen row; any permutation is feature- and score-equivalent."
        ),
    }


@dataclass
class FeatureParityAccumulator:
    feature_names: list[str]
    absolute_tolerance: float = DEFAULT_FEATURE_ATOL

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0:
            raise ValueError("Feature tolerance must be non-negative")
        self.rows = 0
        self.mismatched_rows = 0
        self.mismatch_counts = np.zeros(len(self.feature_names), dtype=np.int64)
        self.maximum_differences = np.zeros(len(self.feature_names), dtype=float)

    def observe(self, product: np.ndarray, reference: np.ndarray) -> None:
        product_values = np.asarray(product, dtype=np.float32)
        reference_values = np.asarray(reference, dtype=np.float32)
        if product_values.shape != reference_values.shape:
            raise ValueError(
                f"Feature chunk shape mismatch: {product_values.shape} vs "
                f"{reference_values.shape}"
            )
        if product_values.ndim != 2 or product_values.shape[1] != len(
            self.feature_names
        ):
            raise ValueError("Feature chunk does not match the ordered contract")
        differences = np.abs(
            product_values.astype(np.float64) - reference_values.astype(np.float64)
        )
        mismatches = differences > self.absolute_tolerance
        self.rows += len(product_values)
        self.mismatched_rows += int(np.count_nonzero(mismatches.any(axis=1)))
        self.mismatch_counts += mismatches.sum(axis=0, dtype=np.int64)
        self.maximum_differences = np.maximum(
            self.maximum_differences, differences.max(axis=0, initial=0.0)
        )

    def report(self, temporal_features: Sequence[str]) -> dict[str, Any]:
        per_feature = {
            name: {
                "mismatched_cells": int(self.mismatch_counts[index]),
                "maximum_absolute_difference": float(
                    self.maximum_differences[index]
                ),
            }
            for index, name in enumerate(self.feature_names)
        }
        temporal_indexes = [self.feature_names.index(name) for name in temporal_features]
        return {
            "rows_compared": self.rows,
            "feature_count": len(self.feature_names),
            "mismatched_rows": self.mismatched_rows,
            "mismatched_cells": int(self.mismatch_counts.sum()),
            "maximum_absolute_numeric_difference": float(
                self.maximum_differences.max(initial=0.0)
            ),
            "comparison_dtype": "float32 (the frozen model-matrix contract)",
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": 0.0,
            "per_feature_mismatch_summary": per_feature,
            "all_43_temporal_features_match": bool(
                np.count_nonzero(self.mismatch_counts[temporal_indexes]) == 0
            ),
        }


def compare_features_and_scores(
    connection: duckdb.DuckDBPyConnection,
    *,
    batch_features: pd.DataFrame,
    builder_source_rows: np.ndarray,
    validation_to_source: np.ndarray,
    contract: FeatureContract,
    bundle: FrozenModelBundle,
    frozen_features_path: Path,
    reference_scores_path: Path,
    feature_atol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream frozen rows while reading product rows by proven source mapping."""

    row_count = len(batch_features)
    if row_count != EXPECTED_ROWS:
        raise AssertionError(f"Product builder returned {row_count:,} rows")
    source_to_builder_position = np.empty(row_count, dtype=np.int64)
    source_to_builder_position[builder_source_rows] = np.arange(
        row_count, dtype=np.int64
    )
    builder_positions = source_to_builder_position[validation_to_source]
    if not np.array_equal(np.sort(builder_positions), np.arange(row_count)):
        raise AssertionError("Builder rows are not a lossless permutation")

    selected = ", ".join(
        f"f.{quote_identifier(name)}" for name in contract.model_features
    )
    connection.execute(
        f"""
        SELECT {selected}, s.context_attack_score::DOUBLE AS reference_attack_score
        FROM (
            SELECT file_row_number::BIGINT AS validation_row,
                   * EXCLUDE (file_row_number)
            FROM read_parquet(
                {sql_string(frozen_features_path)},
                file_row_number = true
            )
        ) f
        JOIN read_parquet({sql_string(reference_scores_path)}) s
          USING (validation_row)
        ORDER BY f.validation_row
        """
    )

    feature_accumulator = FeatureParityAccumulator(
        contract.model_features, absolute_tolerance=feature_atol
    )
    product_scores = np.empty(row_count, dtype=float)
    reference_scores = np.empty(row_count, dtype=float)
    offset = 0
    while True:
        reference_chunk = connection.fetch_df_chunk(20)
        if reference_chunk.empty:
            break
        stop = offset + len(reference_chunk)
        positions = builder_positions[offset:stop]
        product_chunk = batch_features.iloc[positions].to_numpy(
            dtype=np.float32, copy=True
        )
        frozen_chunk = reference_chunk[contract.model_features].to_numpy(
            dtype=np.float32, copy=True
        )
        feature_accumulator.observe(product_chunk, frozen_chunk)
        product_scores[offset:stop] = bundle.predict_scores(product_chunk)
        reference_scores[offset:stop] = reference_chunk[
            "reference_attack_score"
        ].to_numpy(dtype=float)
        offset = stop
    if offset != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS:,} reference rows, read {offset:,}")

    score_differences = np.abs(product_scores - reference_scores)
    feature_report = feature_accumulator.report(contract.temporal_features)
    score_report = {
        "rows_scored": offset,
        "maximum_absolute_score_difference": float(
            score_differences.max(initial=0.0)
        ),
        "score_differences_gt_1e_12": int(
            np.count_nonzero(score_differences > 1e-12)
        ),
        "threshold": FLOW_THRESHOLD,
        "threshold_0_10_decision_disagreements": int(
            np.count_nonzero(
                (product_scores >= FLOW_THRESHOLD)
                != (reference_scores >= FLOW_THRESHOLD)
            )
        ),
    }
    return feature_report, score_report


def validate_frozen_feature_schema(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    contract: FeatureContract,
) -> list[str]:
    described = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(path)})"
    ).fetchall()
    columns = [row[0] for row in described]
    frozen_model_columns = columns[: len(contract.model_features)]
    if frozen_model_columns != contract.model_features:
        raise AssertionError(
            "Frozen research feature ordering differs from cicflow-v2-128"
        )
    return columns


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for path, expected, label in (
        (args.raw_feb17, EXPECTED_RAW_FEB17_SHA256, "isolated raw Feb 17 CSV"),
        (
            args.frozen_features,
            EXPECTED_FROZEN_FEATURES_SHA256,
            "frozen Feb 17 research features",
        ),
        (
            args.reference_scores,
            EXPECTED_REFERENCE_SCORES_SHA256,
            "frozen Feb 17 reference scores",
        ),
    ):
        verify_file(path, expected, label)

    contract = FeatureContract.load(args.contract)
    bundle = FrozenModelBundle.load(args.model, args.manifest, args.contract)
    print("Loading isolated raw Feb 17 flows...", flush=True)
    raw = load_feb17_only_csv(args.raw_feb17)
    raw, removed_evaluation_columns = remove_evaluation_columns(raw, contract)
    product_input_columns = list(raw.columns)
    print(
        f"Building {len(raw):,} label-free product feature rows...", flush=True
    )
    builder = CausalTemporalFeatureBuilder(contract)
    batch = builder.build(raw)
    if not np.array_equal(
        np.sort(batch.identities["source_row_id"].to_numpy(dtype=np.int64)),
        np.arange(EXPECTED_ROWS, dtype=np.int64),
    ):
        raise AssertionError("Product builder did not preserve every source row exactly")
    if list(batch.features.columns) != contract.model_features:
        raise AssertionError("Product feature order differs from cicflow-v2-128")

    raw[PARITY_SOURCE_ROW] = np.arange(EXPECTED_ROWS, dtype=np.int64)
    connection = duckdb.connect()
    try:
        configure_connection(
            connection,
            memory_limit=args.memory_limit,
            threads=args.threads,
            temp_directory=DEFAULT_TEMP_DIRECTORY,
        )
        connection.register("alignment_input", raw)
        validate_frozen_feature_schema(connection, args.frozen_features, contract)
        print("Proving source-row alignment independently of product features...", flush=True)
        validation_to_source, alignment_report = build_lossless_alignment(
            connection,
            frozen_features_path=args.frozen_features,
            contract=contract,
        )
        connection.unregister("alignment_input")
        del raw
        gc.collect()
        connection.execute("DROP TABLE research_generated")
        connection.execute("DROP TABLE research_keyed")
        connection.execute("DROP TABLE frozen_keyed")
        connection.execute("DROP TABLE frozen_rows")
        print("Comparing all 128 features and scoring the product matrix...", flush=True)
        feature_report, score_report = compare_features_and_scores(
            connection,
            batch_features=batch.features,
            builder_source_rows=batch.identities["source_row_id"].to_numpy(
                dtype=np.int64
            ),
            validation_to_source=validation_to_source,
            contract=contract,
            bundle=bundle,
            frozen_features_path=args.frozen_features,
            reference_scores_path=args.reference_scores,
            feature_atol=args.feature_atol,
        )
    finally:
        connection.close()

    result = {
        "verification": "full label-free product feature + frozen model parity",
        "capture_date": CAPTURE_DATE,
        "raw_input": {
            "path": str(args.raw_feb17),
            "sha256": EXPECTED_RAW_FEB17_SHA256,
            "rows": EXPECTED_ROWS,
            "ground_truth_columns_removed_before_product_builder": (
                removed_evaluation_columns
            ),
            "product_input_column_count": len(product_input_columns),
            "product_input_columns": product_input_columns,
            "evaluation_fields_passed_to_product_builder": False,
        },
        "causal_semantics": {
            "state_mode": batch.state_mode,
            "history_upper_boundary": "strictly earlier than current timestamp",
            "same_timestamp_policy": "EXCLUDE GROUP",
        },
        "alignment": alignment_report,
        "feature_parity": {
            **feature_report,
            "feature_order_matches_cicflow_v2_128": True,
        },
        "score_parity": score_report,
        "model_version": bundle.model_version,
        "feature_contract": contract.version,
        "feature_builder_version": contract.builder_version,
        "model_retrained": False,
        "model_retuned": False,
        "holdout_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    if result["feature_parity"]["mismatched_cells"] != 0:
        raise AssertionError("Full product feature parity failed")
    if not result["feature_parity"]["all_43_temporal_features_match"]:
        raise AssertionError("At least one temporal feature differs")
    if result["score_parity"]["score_differences_gt_1e_12"] != 0:
        raise AssertionError("Full product score parity failed")
    if result["score_parity"]["threshold_0_10_decision_disagreements"] != 0:
        raise AssertionError("Full product decision parity failed")
    if args.report_path:
        write_report(args.report_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
