"""Run the first and only Feb 18 evaluation of the fully frozen v2 pipeline.

The Context model, 0.10 flow threshold, Policy B / five-minute aggregation, and
max-score 0.25 promotion rule are integrity-checked before holdout labels are
loaded.  All later drift and error analysis is diagnostic only and cannot alter
the frozen pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .analyze_incident_aggregation import (
        restore_validation_identities,
        route_validation_rows,
        sha256_file,
    )
    from .analyze_incident_promotion import (
        ATTACK_CATEGORIES,
        FLOW_THRESHOLD,
        INCIDENT_KEYS,
        INCIDENT_RECALL_REQUIREMENT,
        INCIDENT_WINDOW_SECONDS,
        MATCH_CONTEXT_FEATURES,
        build_incident_evidence,
        build_reference_attack_incidents,
        category_metrics,
        expression_by_alias,
        promoted_validation_metrics,
        signature_expression,
        ungated_metrics,
    )
    from .build_temporal_features import (
        CONTEXT_FEATURES,
        STATIC_BEHAVIORAL_FEATURES,
        baseline_features,
        read_raw_columns,
        window_clause,
    )
except ImportError:  # Direct execution: python src/evaluate_v2_feb18_holdout.py
    from analyze_incident_aggregation import (
        restore_validation_identities,
        route_validation_rows,
        sha256_file,
    )
    from analyze_incident_promotion import (
        ATTACK_CATEGORIES,
        FLOW_THRESHOLD,
        INCIDENT_KEYS,
        INCIDENT_RECALL_REQUIREMENT,
        INCIDENT_WINDOW_SECONDS,
        MATCH_CONTEXT_FEATURES,
        build_incident_evidence,
        build_reference_attack_incidents,
        category_metrics,
        expression_by_alias,
        promoted_validation_metrics,
        signature_expression,
        ungated_metrics,
    )
    from build_temporal_features import (
        CONTEXT_FEATURES,
        STATIC_BEHAVIORAL_FEATURES,
        baseline_features,
        read_raw_columns,
        window_clause,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cic_unsw_nb15" / "CICFlowMeter_out.csv"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"
FEATURE_BUILDER_PATH = PROJECT_ROOT / "src" / "build_temporal_features.py"
VALIDATION_FEATURES_PATH = FEATURE_DIR / "validation_features.parquet"
HOLDOUT_FEATURES_PATH = FEATURE_DIR / "locked_holdout_features.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
ABLATION_METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "v2_threshold_results.csv"
INCIDENT_AGGREGATION_MANIFEST_PATH = MODELS_DIR / "v2_incident_aggregation_manifest.json"
SELECTED_PROMOTION_PATH = MODELS_DIR / "v2_incident_promotion_selected_policy.json"
PROMOTION_MANIFEST_PATH = MODELS_DIR / "v2_incident_promotion_manifest.json"
PROMOTION_VALIDATION_PATH = MODELS_DIR / "v2_incident_promotion_validation_metrics.json"
OOF_SCORES_PATH = MODELS_DIR / "v2_temporal_oof_scores.parquet"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"

FLOW_METRICS_PATH = MODELS_DIR / "v2_feb18_flow_metrics.json"
INCIDENT_METRICS_PATH = MODELS_DIR / "v2_feb18_incident_metrics.json"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_feb18_category_metrics.csv"
STAGE_COMPARISON_PATH = MODELS_DIR / "v2_feb18_stage_comparison.csv"
DRIFT_METRICS_PATH = MODELS_DIR / "v2_feb18_drift_metrics.json"
FP_PATTERNS_PATH = MODELS_DIR / "v2_feb18_fp_patterns.csv"
FINAL_MANIFEST_PATH = MODELS_DIR / "v2_feb18_final_evaluation_manifest.json"

CACHE_DIR = FEATURE_DIR / "_feb18_final_evaluation_cache"
ROUTED_HOLDOUT_CSV = CACHE_DIR / "feb18_routed_rows.csv"
HOLDOUT_SCORES_CACHE = CACHE_DIR / "feb18_scores.parquet"
ROUTED_VALIDATION_CSV = CACHE_DIR / "feb17_diagnostic_rows.csv"

HOLDOUT_CAPTURE_DATE = "2015-02-18"
HOLDOUT_CAPTURE_DATE_BYTES = b"18/02/2015"
EXPECTED_HOLDOUT_ROWS = 1_275_429
PROMOTION_SCORE_THRESHOLD = 0.25
TOP_IMPORTANT_FEATURES = 10
CATASTROPHIC_RATE_MULTIPLIER = 2.0
METRIC_STABILITY_TOLERANCE = 0.001

# Recorded before the first fully frozen Feb 18 evaluation.  These constants
# turn the preflight into a comparison against an immutable snapshot rather
# than a post-hoc inventory of whatever happens to be on disk.
EXPECTED_FROZEN_SHA256 = {
    "frozen_context_model": (
        "4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72"
    ),
    "preprocessing_feature_manifest": (
        "d313c554798945a5ab34a43237dd0e57019c35e550b2ded987bce7039b5778f8"
    ),
    "preprocessing_builder_code": (
        "7b12e85df31fab6dcb986ea304ae89390dce395078c7b5e232fa6b391be84c18"
    ),
    "threshold_configuration": (
        "868a523fa47c10f98980ec343b98345d66b02abfeea0879e361186e201ae64d8"
    ),
    "incident_aggregation_manifest": (
        "7bbabfa17109ff9921b611b22ca0d6a7f7586075922a561bb766866904134f6a"
    ),
    "selected_promotion_policy": (
        "28def330bfd2742b1ab806d4a1b5a78480c302a6c9309a5c114bbad4f5b073f4"
    ),
    "promotion_manifest": (
        "64761af2a13ba1e19c8b43986d194ca3e7e3fe3a0cf37c45d3f43a67716876f8"
    ),
    "source_dataset": (
        "025a94239f29521e154be3a65d65268404d4c58600b66b94a4235eae87dbd8b5"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep routed raw subsets and temporary holdout scores.",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def locate_timestamp(line: bytes) -> bytes:
    start = 0
    for _ in range(6):
        comma = line.find(b",", start)
        if comma < 0:
            raise ValueError("Malformed CIC CSV row before Timestamp")
        start = comma + 1
    end = line.find(b",", start)
    if end < 0:
        raise ValueError("Malformed CIC CSV row at Timestamp")
    return line[start:end]


def route_holdout_once() -> dict[str, Any]:
    """Materialize Feb 18 source rows exactly once from the combined CSV."""

    started = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ROUTED_HOLDOUT_CSV.with_suffix(".tmp.csv")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    matched = 0
    scanned = 0
    with RAW_PATH.open("rb", buffering=8 * 1024 * 1024) as source, temporary.open(
        "wb", buffering=8 * 1024 * 1024
    ) as target:
        header = source.readline()
        if not header:
            raise ValueError("Raw CIC source is empty")
        columns = header.decode("utf-8-sig").rstrip("\r\n").split(",")
        if len(columns) != 84 or columns[6] != "Timestamp":
            raise AssertionError("Unexpected CIC source schema")
        target.write(header)
        digest.update(header)
        for line in source:
            scanned += 1
            if locate_timestamp(line).startswith(HOLDOUT_CAPTURE_DATE_BYTES):
                target.write(line)
                digest.update(line)
                matched += 1
    if matched != EXPECTED_HOLDOUT_ROWS:
        temporary.unlink(missing_ok=True)
        raise AssertionError(
            f"Expected {EXPECTED_HOLDOUT_ROWS:,} Feb 18 rows, routed {matched:,}"
        )
    os.replace(temporary, ROUTED_HOLDOUT_CSV)
    return {
        "materialization_count": 1,
        "rows": matched,
        "combined_source_rows_scanned": scanned,
        "materialized_non_holdout_rows": 0,
        "sha256": digest.hexdigest(),
        "size_bytes": ROUTED_HOLDOUT_CSV.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
        "source_rows_altered": False,
        "routing_field": "Timestamp only",
    }


def configure_connection(
    connection: duckdb.DuckDBPyConnection, memory_limit: str, threads: int
) -> None:
    connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_string(CACHE_DIR)}")


def preflight_integrity() -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Verify all frozen decisions before any holdout label is loaded."""

    required = (
        RAW_PATH,
        FEATURE_MANIFEST_PATH,
        FEATURE_BUILDER_PATH,
        HOLDOUT_FEATURES_PATH,
        CONTEXT_MODEL_PATH,
        ABLATION_METRICS_PATH,
        THRESHOLD_RESULTS_PATH,
        INCIDENT_AGGREGATION_MANIFEST_PATH,
        SELECTED_PROMOTION_PATH,
        PROMOTION_MANIFEST_PATH,
        PROMOTION_VALIDATION_PATH,
        OOF_SCORES_PATH,
        VALIDATION_SCORES_PATH,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    promotion_manifest = json.loads(PROMOTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    promotion_policy = json.loads(SELECTED_PROMOTION_PATH.read_text(encoding="utf-8"))
    aggregation_manifest = json.loads(
        INCIDENT_AGGREGATION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    ablation = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))

    hashes = {
        "frozen_context_model": sha256_file(CONTEXT_MODEL_PATH),
        "preprocessing_feature_manifest": sha256_file(FEATURE_MANIFEST_PATH),
        "preprocessing_builder_code": sha256_file(FEATURE_BUILDER_PATH),
        "threshold_configuration": sha256_file(ABLATION_METRICS_PATH),
        "threshold_sweep": sha256_file(THRESHOLD_RESULTS_PATH),
        "incident_aggregation_manifest": sha256_file(
            INCIDENT_AGGREGATION_MANIFEST_PATH
        ),
        "selected_promotion_policy": sha256_file(SELECTED_PROMOTION_PATH),
        "promotion_manifest": sha256_file(PROMOTION_MANIFEST_PATH),
        "source_dataset": sha256_file(RAW_PATH),
        "holdout_feature_dataset": sha256_file(HOLDOUT_FEATURES_PATH),
    }
    snapshot_mismatches = {
        name: {"expected": expected, "actual": hashes[name]}
        for name, expected in EXPECTED_FROZEN_SHA256.items()
        if hashes[name] != expected
    }
    if snapshot_mismatches:
        raise AssertionError(
            "Frozen pre-evaluation hash snapshot mismatch: "
            + json.dumps(snapshot_mismatches, sort_keys=True)
        )
    expected_model = promotion_manifest["frozen_components"]["context_model_sha256"]
    expected_preprocessing = promotion_manifest["dataset_hashes"]["feature_manifest"]
    expected_source = promotion_manifest["dataset_hashes"]["combined_raw_source"]
    expected_promotion = promotion_manifest["selected_policy_sha256"]
    if hashes["frozen_context_model"] != expected_model:
        raise AssertionError("Frozen Context model hash mismatch")
    if hashes["preprocessing_feature_manifest"] != expected_preprocessing:
        raise AssertionError("Frozen preprocessing/feature-contract hash mismatch")
    if hashes["source_dataset"] != expected_source:
        raise AssertionError("Raw source dataset hash mismatch")
    if hashes["selected_promotion_policy"] != expected_promotion:
        raise AssertionError("Selected promotion policy hash mismatch")

    ablation_context = ablation["models"]["context_v2"]
    if ablation_context["continuous_metrics"]["model_sha256"] != expected_model:
        raise AssertionError("Ablation model hash disagrees with promotion freeze")
    if not np.isclose(
        ablation_context["selected_operating_point"]["threshold"], FLOW_THRESHOLD
    ):
        raise AssertionError("Frozen flow threshold is not 0.10")
    frozen_aggregation = promotion_manifest["frozen_components"]
    if (
        frozen_aggregation["aggregation_policy"] != "B"
        or frozen_aggregation["grouping_key"] != list(INCIDENT_KEYS)
        or frozen_aggregation["temporal_window_seconds"]
        != INCIDENT_WINDOW_SECONDS
    ):
        raise AssertionError("Frozen incident aggregation configuration changed")
    agg_recommended = aggregation_manifest["candidate_selection"]["recommended"]
    if agg_recommended["policy"] != "B" or agg_recommended["window_seconds"] != 300:
        raise AssertionError("Aggregation manifest does not freeze Policy B / 5m")
    selected_rule = promotion_policy["selected_rule"]
    if (
        selected_rule["family"] != "A"
        or not np.isclose(selected_rule["score_threshold"], PROMOTION_SCORE_THRESHOLD)
        or selected_rule["count_threshold"] is not None
    ):
        raise AssertionError("Frozen incident promotion rule changed")
    if promotion_manifest["scope"]["validation_used_for_rule_selection"]:
        raise AssertionError("Promotion manifest reports validation selection leakage")
    if promotion_manifest["scope"]["locked_holdout_loaded"]:
        raise AssertionError("Promotion manifest reports prior holdout access")

    baseline = baseline_features(read_raw_columns(RAW_PATH))
    context_features = [*baseline, *STATIC_BEHAVIORAL_FEATURES, *CONTEXT_FEATURES]
    if len(baseline) != 76 or len(context_features) != 128:
        raise AssertionError("Frozen v2 feature contract changed")
    preflight = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_labels_loaded": False,
        "hashes": hashes,
        "expected_hashes": {
            **EXPECTED_FROZEN_SHA256,
            "holdout_feature_dataset_from_feature_manifest": json.loads(
                FEATURE_MANIFEST_PATH.read_text(encoding="utf-8")
            )["outputs"]["locked_holdout"]["sha256"],
        },
        "all_expected_hashes_match": True,
        "preprocessing_artifact_note": (
            "V2 uses a deterministic numeric feature pipeline, not a serialized "
            "scaler/encoder. The frozen preprocessing artifacts are "
            "feature_manifest.json and build_temporal_features.py."
        ),
        "frozen_configuration_before_holdout": {
            "flow_threshold": FLOW_THRESHOLD,
            "aggregation_policy": "B",
            "grouping_key": list(INCIDENT_KEYS),
            "aggregation_window_seconds": INCIDENT_WINDOW_SECONDS,
            "promotion_rule": selected_rule,
        },
    }
    return preflight, context_features, promotion_policy


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("output_frame", frame)
        connection.execute(
            f"""
            COPY output_frame TO {sql_string(temporary)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        os.replace(temporary, path)
    finally:
        connection.close()


def score_holdout(
    model: Any,
    context_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the holdout feature rows once and generate frozen Context scores."""

    started = time.perf_counter()
    scores = np.empty(EXPECTED_HOLDOUT_ROWS, dtype=np.float64)
    labels = np.empty(EXPECTED_HOLDOUT_ROWS, dtype=np.uint8)
    categories = np.empty(EXPECTED_HOLDOUT_ROWS, dtype=object)
    selected = ", ".join(quote_identifier(column) for column in context_features)
    connection = duckdb.connect()
    row_offset = 0
    inference_seconds = 0.0
    try:
        connection.execute(
            f"""
            SELECT row_number() OVER () - 1 AS holdout_row,
                   {selected}, label, attack_cat
            FROM read_parquet({sql_string(HOLDOUT_FEATURES_PATH)})
            """
        )
        while True:
            chunk = connection.fetch_df_chunk(100)
            if chunk.empty:
                break
            end = row_offset + len(chunk)
            expected_rows = np.arange(row_offset, end)
            if not np.array_equal(chunk["holdout_row"].to_numpy(), expected_rows):
                raise AssertionError("Holdout Parquet row order changed while scoring")
            matrix = chunk[context_features].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(matrix).all():
                raise AssertionError("Holdout model input contains NaN or infinity")
            inference_started = time.perf_counter()
            scores[row_offset:end] = model.predict_proba(matrix)[:, 1]
            inference_seconds += time.perf_counter() - inference_started
            labels[row_offset:end] = chunk["label"].to_numpy(dtype=np.uint8)
            categories[row_offset:end] = chunk["attack_cat"].astype(str).to_numpy()
            row_offset = end
    finally:
        connection.close()
    if row_offset != EXPECTED_HOLDOUT_ROWS:
        raise AssertionError(f"Scored {row_offset:,} holdout rows")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise AssertionError("Invalid holdout targets or scores")
    score_frame = pd.DataFrame(
        {
            "holdout_row": np.arange(EXPECTED_HOLDOUT_ROWS, dtype=np.int64),
            "attack_cat": categories,
            "label": labels,
            "attack_score": scores,
            "predicted_class": (scores >= FLOW_THRESHOLD).astype(np.uint8),
        }
    )
    write_parquet(score_frame, HOLDOUT_SCORES_CACHE)
    return scores, labels, categories, {
        "rows": row_offset,
        "holdout_feature_loads": 1,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": time.perf_counter() - started,
        "score_cache_sha256": sha256_file(HOLDOUT_SCORES_CACHE),
    }


def verify_validation_score_compatibility(
    model: Any, context_features: list[str]
) -> dict[str, Any]:
    """Compare this runtime with the frozen Feb 17 scores without touching Feb 18."""

    selected = ", ".join(quote_identifier(column) for column in context_features)
    connection = duckdb.connect()
    compared = 0
    maximum_absolute_difference = 0.0
    score_differences = 0
    threshold_disagreements = 0
    try:
        connection.execute(
            f"""
            SELECT {selected}, s.context_attack_score
            FROM (
                SELECT row_number() OVER () - 1 AS validation_row, *
                FROM read_parquet({sql_string(VALIDATION_FEATURES_PATH)})
            ) f
            JOIN read_parquet({sql_string(VALIDATION_SCORES_PATH)}) s
              USING (validation_row)
            ORDER BY validation_row
            """
        )
        while True:
            chunk = connection.fetch_df_chunk(100)
            if chunk.empty:
                break
            matrix = chunk[context_features].to_numpy(dtype=np.float32, copy=True)
            current = model.predict_proba(matrix)[:, 1]
            frozen = chunk["context_attack_score"].to_numpy(dtype=float)
            absolute = np.abs(current - frozen)
            maximum_absolute_difference = max(
                maximum_absolute_difference, float(absolute.max(initial=0.0))
            )
            score_differences += int(np.count_nonzero(absolute > 1e-12))
            threshold_disagreements += int(
                np.count_nonzero(
                    (current >= FLOW_THRESHOLD) != (frozen >= FLOW_THRESHOLD)
                )
            )
            compared += len(chunk)
    finally:
        connection.close()
    if compared != 498_890:
        raise AssertionError(f"Compared {compared:,} Feb 17 runtime-check rows")
    return {
        "comparison_data": "Feb 17 frozen validation only; Feb 18 not accessed",
        "rows_compared": compared,
        "maximum_absolute_score_difference": maximum_absolute_difference,
        "score_differences_gt_1e_12": score_differences,
        "threshold_0_10_class_disagreements": threshold_disagreements,
        "exactly_reproduces_frozen_validation_scores": score_differences == 0,
        "operational_predictions_identical": threshold_disagreements == 0,
    }


def holdout_identity_query(source_relation: str, baseline: list[str]) -> str:
    baseline_sql = ",\n        ".join(quote_identifier(column) for column in baseline)
    expression_map = expression_by_alias()
    context_sql = ",\n        ".join(
        expression_map[column] for column in MATCH_CONTEXT_FEATURES
    )
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


def restore_holdout_identities(
    baseline: list[str], memory_limit: str, threads: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    match_columns = [
        *baseline,
        *STATIC_BEHAVIORAL_FEATURES,
        *MATCH_CONTEXT_FEATURES,
        "attack_cat",
        "label",
    ]
    signature = signature_expression(match_columns)
    connection = duckdb.connect()
    try:
        configure_connection(connection, memory_limit, threads)
        connection.execute(
            f"""
            CREATE TABLE routed_holdout AS
            SELECT *,
                   strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
                   ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE AS flow_bytes,
                   ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
            FROM read_csv_auto(
                {sql_string(ROUTED_HOLDOUT_CSV)}, header=true,
                sample_size=-1, parallel=true
            )
            """
        )
        raw_stats = connection.execute(
            """
            SELECT count(*), min(event_ts), max(event_ts),
                   count(*) FILTER (WHERE CAST(event_ts AS DATE) != DATE '2015-02-18')
            FROM routed_holdout
            """
        ).fetchone()
        if raw_stats is None or tuple(map(int, (raw_stats[0], raw_stats[3]))) != (
            EXPECTED_HOLDOUT_ROWS,
            0,
        ):
            raise AssertionError(f"Invalid routed holdout rows: {raw_stats}")
        connection.execute(
            f"CREATE TABLE identity_features AS {holdout_identity_query('routed_holdout', baseline)}"
        )
        connection.execute(
            f"""
            CREATE TABLE frozen_holdout_rows AS
            SELECT row_number() OVER () - 1 AS holdout_row, *
            FROM read_parquet({sql_string(HOLDOUT_FEATURES_PATH)})
            """
        )
        connection.execute(
            f"""
            CREATE TABLE frozen_keyed AS
            SELECT *, row_number() OVER (
                PARTITION BY signature ORDER BY holdout_row
            ) AS duplicate_rank
            FROM (SELECT *, {signature} AS signature FROM frozen_holdout_rows)
            """
        )
        connection.execute(
            f"""
            CREATE TABLE identity_keyed AS
            SELECT *, row_number() OVER (
                PARTITION BY signature ORDER BY timestamp, flow_id, src_ip, dst_ip
            ) AS duplicate_rank
            FROM (SELECT *, {signature} AS signature FROM identity_features)
            """
        )
        mismatch = " OR ".join(
            f"(f.{quote_identifier(column)} IS DISTINCT FROM i.{quote_identifier(column)})"
            for column in match_columns
        )
        alignment = connection.execute(
            f"""
            SELECT count(*), count(*) FILTER (WHERE {mismatch}),
                   count(DISTINCT f.holdout_row),
                   count(*) FILTER (
                       WHERE s.label IS DISTINCT FROM f.label
                          OR s.attack_cat IS DISTINCT FROM f.attack_cat
                   )
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(HOLDOUT_SCORES_CACHE)}) s USING (holdout_row)
            """
        ).fetchone()
        if alignment is None or tuple(map(int, alignment)) != (
            EXPECTED_HOLDOUT_ROWS,
            0,
            EXPECTED_HOLDOUT_ROWS,
            0,
        ):
            raise AssertionError(f"Holdout identity alignment failed: {alignment}")
        relevant = connection.execute(
            f"""
            SELECT
                s.holdout_row AS validation_row,
                i.timestamp,
                i.flow_id,
                i.src_ip,
                i.dst_ip,
                i."Src Port"::BIGINT AS src_port,
                i."Dst Port"::BIGINT AS dst_port,
                i."Protocol"::BIGINT AS protocol,
                s.attack_cat,
                s.label::UTINYINT AS label,
                s.attack_score::DOUBLE AS attack_score,
                s.predicted_class::UTINYINT AS predicted_class
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(HOLDOUT_SCORES_CACHE)}) s USING (holdout_row)
            WHERE s.label = 1 OR s.predicted_class = 1
            ORDER BY s.holdout_row
            """
        ).fetchdf()
    finally:
        connection.close()
    return relevant, {
        "matched_rows": int(alignment[0]),
        "feature_value_mismatches": int(alignment[1]),
        "score_target_mismatches": int(alignment[3]),
        "minimum_timestamp": raw_stats[1].isoformat(),
        "maximum_timestamp": raw_stats[2].isoformat(),
        "materialized_relevant_rows": len(relevant),
        "match_signature_columns": len(match_columns),
        "runtime_seconds": time.perf_counter() - started,
    }


def binary_flow_metrics(
    labels: np.ndarray, scores: np.ndarray, capture_hours: float
) -> dict[str, Any]:
    predictions = (scores >= FLOW_THRESHOLD).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    return {
        "rows": len(labels),
        "normal_flows": int((labels == 0).sum()),
        "attack_flows": int((labels == 1).sum()),
        "threshold": FLOW_THRESHOLD,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc_average_precision": float(average_precision_score(labels, scores)),
        "alerts": int(predictions.sum()),
        "capture_hours": capture_hours,
        "alerts_per_hour": float(predictions.sum() / capture_hours),
        "fp_flows_per_hour": float(fp / capture_hours),
        "detected_attacks_per_hour": float(tp / capture_hours),
        "missed_attacks_per_hour": float(fn / capture_hours),
    }


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(quantiles) < 2:
        return 0.0 if np.all(current == reference[0]) else float("inf")
    edges = quantiles.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    epsilon = 1e-6
    reference_share = np.clip(reference_counts / len(reference), epsilon, None)
    current_share = np.clip(current_counts / len(current), epsilon, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def categorical_frequencies(values: Iterable[Any]) -> dict[str, float]:
    series = pd.Series(values, dtype="string").fillna("<NA>")
    return {str(key): float(value) for key, value in series.value_counts(normalize=True).items()}


def jensen_shannon_divergence(
    reference: dict[str, float], current: dict[str, float]
) -> float:
    levels = sorted(set(reference) | set(current))
    p = np.array([reference.get(level, 0.0) for level in levels], dtype=float)
    q = np.array([current.get(level, 0.0) for level in levels], dtype=float)
    midpoint = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_positive = p > 0
    q_positive = q > 0
    p_term[p_positive] = p[p_positive] * np.log2(
        p[p_positive] / midpoint[p_positive]
    )
    q_term[q_positive] = q[q_positive] * np.log2(
        q[q_positive] / midpoint[q_positive]
    )
    return float(0.5 * (p_term.sum() + q_term.sum()))


def classify_temporal_changes(changes: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Label comparable Feb18-minus-Feb17 deltas without altering the pipeline."""

    lower_is_better = {
        "flow_fpr",
        "aggregation_fp_incidents_per_hour",
        "aggregation_total_incidents_per_hour",
        "promotion_fp_incidents_per_hour",
        "promotion_total_incidents_per_hour",
    }
    output: dict[str, dict[str, Any]] = {}
    for metric, delta in changes.items():
        if abs(delta) <= METRIC_STABILITY_TOLERANCE:
            status = "stable"
        elif metric in lower_is_better:
            status = "improved" if delta < 0 else "degraded"
        else:
            status = "improved" if delta > 0 else "degraded"
        output[metric] = {
            "absolute_change_feb18_minus_feb17": float(delta),
            "status": status,
        }
    return output


def categorical_drift(reference: Iterable[Any], current: Iterable[Any]) -> dict[str, Any]:
    reference_freq = categorical_frequencies(reference)
    current_freq = categorical_frequencies(current)
    unseen = sorted(set(current_freq) - set(reference_freq))
    return {
        "reference_top_frequencies": dict(list(reference_freq.items())[:20]),
        "current_top_frequencies": dict(list(current_freq.items())[:20]),
        "unseen_level_count": len(unseen),
        "unseen_levels_sample": unseen[:100],
        "jensen_shannon_divergence_bits": jensen_shannon_divergence(
            reference_freq, current_freq
        ),
    }


def load_feature_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    selected = ", ".join(quote_identifier(column) for column in columns)
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"SELECT {selected} FROM read_parquet({sql_string(path)})"
        ).fetchdf()
    finally:
        connection.close()


def calculate_drift(
    model: Any,
    context_features: list[str],
    validation_scores: np.ndarray,
    holdout_scores: np.ndarray,
    validation_evidence: pd.DataFrame,
    holdout_evidence: pd.DataFrame,
) -> dict[str, Any]:
    importances = np.asarray(model.feature_importances_, dtype=float)
    top_indices = np.argsort(importances)[::-1][:TOP_IMPORTANT_FEATURES]
    top_features = [context_features[index] for index in top_indices]
    load_columns = list(dict.fromkeys([*top_features, "Protocol", "Dst Port"]))
    validation_values = load_feature_columns(VALIDATION_FEATURES_PATH, load_columns)
    holdout_values = load_feature_columns(HOLDOUT_FEATURES_PATH, load_columns)

    numeric = []
    for feature, index in zip(top_features, top_indices, strict=True):
        reference = validation_values[feature].to_numpy(dtype=float)
        current = holdout_values[feature].to_numpy(dtype=float)
        numeric.append(
            {
                "feature": feature,
                "random_forest_importance": float(importances[index]),
                "feb17": numeric_summary(reference),
                "feb18": numeric_summary(current),
                "psi_feb17_reference_bins": population_stability_index(
                    reference, current
                ),
            }
        )

    validation_protocol = validation_values["Protocol"].astype("Int64").astype(str)
    holdout_protocol = holdout_values["Protocol"].astype("Int64").astype(str)
    validation_service = (
        validation_protocol
        + "/"
        + validation_values["Dst Port"].astype("Int64").astype(str)
    )
    holdout_service = (
        holdout_protocol
        + "/"
        + holdout_values["Dst Port"].astype("Int64").astype(str)
    )

    incident_drift: dict[str, Any] = {}
    for display_name, column in (
        ("alert_flow_count", "alert_flow_count"),
        ("max_attack_score", "max_attack_score"),
        ("incident_duration_seconds", "duration_seconds"),
        ("flows_per_incident", "alert_flow_count"),
    ):
        reference = validation_evidence[column].to_numpy(dtype=float)
        current = holdout_evidence[column].to_numpy(dtype=float)
        incident_drift[display_name] = {
            "feb17": numeric_summary(reference),
            "feb18": numeric_summary(current),
            "psi_feb17_reference_bins": population_stability_index(reference, current),
        }
    return {
        "diagnostic_only": True,
        "pipeline_modified_from_diagnostics": False,
        "numeric_drift_measure": (
            "PSI with decile bins learned from Feb 17; duplicate quantile edges removed"
        ),
        "categorical_drift_measure": "Jensen-Shannon divergence in bits",
        "context_attack_score": {
            "feb17": numeric_summary(validation_scores),
            "feb18": numeric_summary(holdout_scores),
            "psi_feb17_reference_bins": population_stability_index(
                validation_scores, holdout_scores
            ),
        },
        "categorical": {
            "proto": {
                "source_column": "Protocol",
                **categorical_drift(validation_protocol, holdout_protocol),
            },
            "service": {
                "source_column": "Protocol/Dst Port diagnostic proxy",
                "explicit_service_column_available": False,
                **categorical_drift(validation_service, holdout_service),
            },
            "state": {
                "available": False,
                "reason": (
                    "CICFlowMeter_out.csv has TCP flag counts but no UNSW state "
                    "field; no state proxy was invented after holdout access."
                ),
            },
        },
        "top_random_forest_numeric_features": numeric,
        "incident_evidence": incident_drift,
    }


def pure_fp_patterns(
    flows: pd.DataFrame,
    alert_membership: pd.DataFrame,
    incident_truth: pd.DataFrame,
    promoted_ids: set[str],
    flow_fp_count: int,
    stage: str,
) -> pd.DataFrame:
    pure_ids = set(
        incident_truth.loc[incident_truth["pure_false_positive"], "incident_id"]
    )
    if stage == "promoted":
        pure_ids &= promoted_ids
    fp_members = alert_membership.loc[
        alert_membership["incident_id"].isin(pure_ids)
    ].merge(
        flows[
            [
                "validation_row",
                "src_ip",
                "dst_ip",
                "dst_port",
                "protocol",
                "label",
            ]
        ],
        on="validation_row",
        validate="1:1",
    )
    if not (fp_members["label"] == 0).all():
        raise AssertionError("Pure-FP pattern input contains an attack flow")
    pattern_columns = ["src_ip", "dst_ip", "dst_port", "protocol"]
    grouped = (
        fp_members.groupby(pattern_columns, observed=True, dropna=False)
        .agg(
            fp_flows=("validation_row", "size"),
            fp_incidents=("incident_id", "nunique"),
        )
        .reset_index()
        .sort_values(["fp_flows", "fp_incidents"], ascending=[False, False])
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["stage"] = stage
    grouped["percentage_all_fp_flows"] = grouped["fp_flows"] / flow_fp_count
    grouped["percentage_all_fp_incidents"] = grouped["fp_incidents"] / len(pure_ids)
    grouped["pattern"] = (
        grouped["src_ip"].astype(str)
        + "|"
        + grouped["dst_ip"].astype(str)
        + "|"
        + grouped["dst_port"].astype(str)
        + "|"
        + grouped["protocol"].astype(str)
    )
    return grouped.head(20).reset_index(drop=True)


def add_feb17_pattern_comparison(
    holdout_patterns: pd.DataFrame,
    validation_patterns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = holdout_patterns.copy()
    output["appeared_in_feb17_top20"] = False
    summaries: dict[str, Any] = {}
    for stage in ("aggregation", "promoted"):
        validation_keys = set(
            validation_patterns.loc[validation_patterns["stage"] == stage, "pattern"]
        )
        mask = output["stage"] == stage
        output.loc[mask, "appeared_in_feb17_top20"] = output.loc[
            mask, "pattern"
        ].isin(validation_keys)
        current = output.loc[mask]
        summaries[stage] = {
            "top20_exact_pattern_overlap_count": int(
                current["appeared_in_feb17_top20"].sum()
            ),
            "feb18_top20_ports": {
                str(key): int(value)
                for key, value in current.groupby("dst_port")["fp_flows"]
                .sum()
                .sort_values(ascending=False)
                .items()
            },
            "feb18_top20_protocols": {
                str(key): int(value)
                for key, value in current.groupby("protocol")["fp_flows"]
                .sum()
                .sort_values(ascending=False)
                .items()
            },
        }
    return output, summaries


def category_output(
    flows: pd.DataFrame,
    alert_membership: pd.DataFrame,
    promoted_ids: set[str],
) -> pd.DataFrame:
    metrics = category_metrics(
        "holdout_feb18", flows, alert_membership, promoted_ids
    )
    result = metrics.rename(
        columns={
            "attack_flows": "actual_attack_flows",
            "detected_attack_flows_at_flow_threshold": "detected_attack_flows",
            "reference_attack_incidents": "reference_attack_incidents",
            "promoted_detected_attack_incidents": "detected_promoted_attack_incidents",
            "promoted_missed_attack_incidents": "incident_false_negatives",
            "promoted_incident_recall": "incident_recall",
        }
    )
    result["flow_false_negatives"] = (
        result["actual_attack_flows"] - result["detected_attack_flows"]
    )
    return result[
        [
            "attack_category",
            "actual_attack_flows",
            "detected_attack_flows",
            "flow_false_negatives",
            "flow_recall",
            "reference_attack_incidents",
            "ungated_detected_attack_incidents",
            "ungated_incident_recall",
            "detected_promoted_attack_incidents",
            "incident_false_negatives",
            "incident_recall",
            "sufficient_ground_truth",
        ]
    ]


def stage_row(
    split: str,
    stage: str,
    metric_level: str,
    volume: int,
    fp_volume: int,
    recall: float,
    precision: float,
    rate_per_hour: float,
    fp_rate_per_hour: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "stage": stage,
        "metric_level": metric_level,
        "volume": volume,
        "fp_volume": fp_volume,
        "recall": recall,
        "precision": precision,
        "rate_per_hour": rate_per_hour,
        "fp_rate_per_hour": fp_rate_per_hour,
    }


def build_temporal_comparison(
    holdout_flow: dict[str, Any],
    holdout_incident: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ablation = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    feb17_promotion = json.loads(PROMOTION_VALIDATION_PATH.read_text(encoding="utf-8"))
    jan_promotion = json.loads(SELECTED_PROMOTION_PATH.read_text(encoding="utf-8"))

    connection = duckdb.connect()
    try:
        oof = connection.execute(
            f"SELECT label, context_score FROM read_parquet({sql_string(OOF_SCORES_PATH)})"
        ).fetchnumpy()
    finally:
        connection.close()
    jan_hours = (
        jan_promotion["ungated_oof_metrics"]["total_incidents"]
        / jan_promotion["ungated_oof_metrics"]["incidents_per_hour"]
    )
    jan_flow = binary_flow_metrics(
        np.asarray(oof["label"], dtype=np.uint8),
        np.asarray(oof["context_score"], dtype=float),
        jan_hours,
    )
    feb17_selected = ablation["models"]["context_v2"]["selected_operating_point"]
    feb17_continuous = ablation["models"]["context_v2"]["continuous_metrics"]
    feb17_before = feb17_promotion["before_promotion"]
    feb17_after = feb17_promotion["after_promotion"]
    feb17_hours = feb17_promotion["capture_hours_inclusive"]
    feb17_flow = {
        "alerts": feb17_selected["true_positives"] + feb17_selected["false_positives"],
        "false_positives": feb17_selected["false_positives"],
        "recall": feb17_selected["recall"],
        "precision": feb17_selected["precision"],
        "fpr": feb17_selected["fpr"],
        "pr_auc_average_precision": feb17_continuous["pr_auc_average_precision"],
        "alerts_per_hour": (
            feb17_selected["true_positives"] + feb17_selected["false_positives"]
        )
        / feb17_hours,
        "fp_flows_per_hour": feb17_selected["false_positives"] / feb17_hours,
    }
    jan_before = jan_promotion["ungated_oof_metrics"]
    jan_after = jan_promotion["selected_oof_metrics"]
    rows = [
        stage_row(
            "jan22_oof", "raw_flow_detector", "flow", jan_flow["alerts"],
            jan_flow["false_positives"], jan_flow["recall"], jan_flow["precision"],
            jan_flow["alerts_per_hour"], jan_flow["fp_flows_per_hour"],
        ),
        stage_row(
            "jan22_oof", "policy_b_5m_aggregation", "incident",
            jan_before["total_incidents"], jan_before["pure_fp_incidents"],
            jan_before["incident_recall"], jan_before["incident_precision"],
            jan_before["incidents_per_hour"], jan_before["fp_incidents_per_hour"],
        ),
        stage_row(
            "jan22_oof", "policy_b_5m_plus_promotion", "incident",
            jan_after["promoted_total_incidents"], jan_after["promoted_pure_fp_incidents"],
            jan_after["incident_recall"], jan_after["incident_precision"],
            jan_after["total_promoted_incidents_per_hour"], jan_after["fp_incidents_per_hour"],
        ),
        stage_row(
            "feb17_validation", "raw_flow_detector", "flow", feb17_flow["alerts"],
            feb17_flow["false_positives"], feb17_flow["recall"], feb17_flow["precision"],
            feb17_flow["alerts_per_hour"], feb17_flow["fp_flows_per_hour"],
        ),
        stage_row(
            "feb17_validation", "policy_b_5m_aggregation", "incident",
            feb17_before["total_incidents"], feb17_before["pure_fp_incidents"],
            feb17_before["incident_recall"], feb17_before["incident_precision"],
            feb17_before["incidents_per_hour"], feb17_before["fp_incidents_per_hour"],
        ),
        stage_row(
            "feb17_validation", "policy_b_5m_plus_promotion", "incident",
            feb17_after["total_promoted_incidents"], feb17_after["pure_fp_promoted_incidents"],
            feb17_after["incident_recall"], feb17_after["incident_precision"],
            feb17_after["incidents_per_hour"], feb17_after["fp_incidents_per_hour"],
        ),
        stage_row(
            "feb18_holdout", "raw_flow_detector", "flow", holdout_flow["alerts"],
            holdout_flow["false_positives"], holdout_flow["recall"], holdout_flow["precision"],
            holdout_flow["alerts_per_hour"], holdout_flow["fp_flows_per_hour"],
        ),
        stage_row(
            "feb18_holdout", "policy_b_5m_aggregation", "incident",
            holdout_incident["before_promotion"]["total_incidents"],
            holdout_incident["before_promotion"]["pure_fp_incidents"],
            holdout_incident["before_promotion"]["incident_recall"],
            holdout_incident["before_promotion"]["incident_precision"],
            holdout_incident["before_promotion"]["incidents_per_hour"],
            holdout_incident["before_promotion"]["fp_incidents_per_hour"],
        ),
        stage_row(
            "feb18_holdout", "policy_b_5m_plus_promotion", "incident",
            holdout_incident["after_promotion"]["total_promoted_incidents"],
            holdout_incident["after_promotion"]["pure_fp_promoted_incidents"],
            holdout_incident["after_promotion"]["incident_recall"],
            holdout_incident["after_promotion"]["incident_precision"],
            holdout_incident["after_promotion"]["incidents_per_hour"],
            holdout_incident["after_promotion"]["fp_incidents_per_hour"],
        ),
    ]
    comparison = pd.DataFrame(rows)

    temporal = {
        "flow": {
            "jan22_oof": {
                key: jan_flow[key] for key in ("recall", "fpr", "pr_auc_average_precision")
            },
            "feb17_validation": {
                key: feb17_flow[key] for key in ("recall", "fpr", "pr_auc_average_precision")
            },
            "feb18_holdout": {
                key: holdout_flow[key] for key in ("recall", "fpr", "pr_auc_average_precision")
            },
        },
        "incident_aggregation": {
            "jan22_oof": jan_before,
            "feb17_validation": feb17_before,
            "feb18_holdout": holdout_incident["before_promotion"],
        },
        "promoted_pipeline": {
            "jan22_oof": jan_after,
            "feb17_validation": feb17_after,
            "feb18_holdout": holdout_incident["after_promotion"],
        },
    }
    temporal["absolute_changes_feb18_minus_feb17"] = {
        "flow_recall": holdout_flow["recall"] - feb17_flow["recall"],
        "flow_fpr": holdout_flow["fpr"] - feb17_flow["fpr"],
        "flow_pr_auc": holdout_flow["pr_auc_average_precision"]
        - feb17_flow["pr_auc_average_precision"],
        "aggregation_incident_recall": holdout_incident["before_promotion"]["incident_recall"]
        - feb17_before["incident_recall"],
        "aggregation_fp_incidents_per_hour": holdout_incident["before_promotion"]["fp_incidents_per_hour"]
        - feb17_before["fp_incidents_per_hour"],
        "aggregation_total_incidents_per_hour": holdout_incident["before_promotion"]["incidents_per_hour"]
        - feb17_before["incidents_per_hour"],
        "promotion_incident_recall": holdout_incident["after_promotion"]["incident_recall"]
        - feb17_after["incident_recall"],
        "promotion_incident_precision": holdout_incident["after_promotion"]["incident_precision"]
        - feb17_after["incident_precision"],
        "promotion_fp_incidents_per_hour": holdout_incident["after_promotion"]["fp_incidents_per_hour"]
        - feb17_after["fp_incidents_per_hour"],
        "promotion_total_incidents_per_hour": holdout_incident["after_promotion"]["incidents_per_hour"]
        - feb17_after["incidents_per_hour"],
    }
    temporal["metric_status_feb18_vs_feb17"] = classify_temporal_changes(
        temporal["absolute_changes_feb18_minus_feb17"]
    )
    temporal["stability_rule"] = (
        f"Absolute deltas <= {METRIC_STABILITY_TOLERANCE} are labeled stable; "
        "otherwise direction is judged by the metric's operational meaning."
    )
    return comparison, temporal


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    total_started = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying frozen pipeline integrity before holdout label access...", flush=True)
    preflight, context_features, promotion_policy = preflight_integrity()
    preflight_hash_snapshot = dict(preflight["hashes"])
    baseline = context_features[:76]
    selected_rule = promotion_policy["selected_rule"]

    model = joblib.load(CONTEXT_MODEL_PATH)
    if list(model.classes_) != [0, 1] or not hasattr(model, "feature_importances_"):
        raise AssertionError("Frozen Context model interface changed")
    runtime_compatibility = verify_validation_score_compatibility(
        model, context_features
    )
    if not runtime_compatibility["operational_predictions_identical"]:
        raise AssertionError(
            "Current runtime does not reproduce frozen Feb 17 predictions"
        )

    # First and only holdout materialization and score evaluation.
    print("Materializing the Feb 18 source subset exactly once...", flush=True)
    materialization = route_holdout_once()
    print("Scoring Feb 18 with the frozen Context model...", flush=True)
    scores, labels, categories, scoring = score_holdout(model, context_features)
    print("Restoring Feb 18 identities for frozen Policy B aggregation...", flush=True)
    holdout_flows, identity_alignment = restore_holdout_identities(
        baseline, args.memory_limit, args.threads
    )
    minimum_timestamp = pd.Timestamp(identity_alignment["minimum_timestamp"])
    maximum_timestamp = pd.Timestamp(identity_alignment["maximum_timestamp"])
    capture_seconds = (maximum_timestamp - minimum_timestamp).total_seconds() + 1.0
    capture_hours = capture_seconds / 3600.0
    flow_metrics = binary_flow_metrics(labels, scores, capture_hours)
    flow_metrics.update(
        {
            "capture_date": HOLDOUT_CAPTURE_DATE,
            "minimum_timestamp": minimum_timestamp.isoformat(),
            "maximum_timestamp": maximum_timestamp.isoformat(),
            "capture_seconds_inclusive": capture_seconds,
            "flows_per_hour": len(labels) / capture_hours,
            "model_inference_seconds": scoring["inference_seconds"],
            "feb17_reference": {
                "recall": 0.9859078856918393,
                "fpr": 0.02525474166394998,
                "false_positives": 12_085,
                "false_negatives": 287,
                "pr_auc_average_precision": 0.8477457428319373,
            },
        }
    )
    flow_metrics["feb18_minus_feb17"] = {
        "recall": flow_metrics["recall"]
        - flow_metrics["feb17_reference"]["recall"],
        "fpr": flow_metrics["fpr"] - flow_metrics["feb17_reference"]["fpr"],
        "pr_auc_average_precision": flow_metrics["pr_auc_average_precision"]
        - flow_metrics["feb17_reference"]["pr_auc_average_precision"],
        "fp_flows_per_hour": flow_metrics["fp_flows_per_hour"]
        - 12_085 / (12_993 / 3600),
    }
    FLOW_METRICS_PATH.write_text(
        json.dumps(json_ready(flow_metrics), indent=2), encoding="utf-8"
    )

    evidence, alert_membership, incident_truth = build_incident_evidence(holdout_flows)
    reference_incidents = build_reference_attack_incidents(holdout_flows)
    incident_metrics, promoted_ids = promoted_validation_metrics(
        evidence,
        alert_membership,
        incident_truth,
        reference_incidents,
        selected_rule,
        capture_hours,
    )
    before = incident_metrics["before_promotion"]
    after = incident_metrics["after_promotion"]
    incident_metrics.update(
        {
            "flow_alerts": flow_metrics["alerts"],
            "flow_false_positives": flow_metrics["false_positives"],
            "fp_flow_compression_ratio": flow_metrics["false_positives"]
            / before["pure_fp_incidents"],
            "alert_to_incident_reduction_percentage": 1.0
            - before["total_incidents"] / flow_metrics["alerts"],
            "promotion_rule": selected_rule,
            "changes_after_promotion_vs_flow_alerts": {
                "volume_absolute": flow_metrics["alerts"]
                - after["total_promoted_incidents"],
                "volume_percentage": 1.0
                - after["total_promoted_incidents"] / flow_metrics["alerts"],
                "fp_volume_absolute": flow_metrics["false_positives"]
                - after["pure_fp_promoted_incidents"],
                "fp_volume_percentage": 1.0
                - after["pure_fp_promoted_incidents"]
                / flow_metrics["false_positives"],
                "unit_note": "flow alerts versus promoted incidents",
            },
            "changes_after_promotion_vs_aggregation": incident_metrics["reductions"],
        }
    )
    feb17_promotion = json.loads(PROMOTION_VALIDATION_PATH.read_text(encoding="utf-8"))
    feb17_after = feb17_promotion["after_promotion"]
    incident_metrics["changes_after_promotion_vs_feb17_promoted"] = {
        "total_incidents_absolute": after["total_promoted_incidents"]
        - feb17_after["total_promoted_incidents"],
        "total_incidents_percentage": after["total_promoted_incidents"]
        / feb17_after["total_promoted_incidents"]
        - 1.0,
        "pure_fp_incidents_absolute": after["pure_fp_promoted_incidents"]
        - feb17_after["pure_fp_promoted_incidents"],
        "pure_fp_incidents_percentage": after["pure_fp_promoted_incidents"]
        / feb17_after["pure_fp_promoted_incidents"]
        - 1.0,
        "incident_recall_absolute": after["incident_recall"]
        - feb17_after["incident_recall"],
        "incident_precision_absolute": after["incident_precision"]
        - feb17_after["incident_precision"],
        "promoted_incidents_per_hour_absolute": after["incidents_per_hour"]
        - feb17_after["incidents_per_hour"],
        "fp_promoted_incidents_per_hour_absolute": after["fp_incidents_per_hour"]
        - feb17_after["fp_incidents_per_hour"],
    }
    INCIDENT_METRICS_PATH.write_text(
        json.dumps(json_ready(incident_metrics), indent=2), encoding="utf-8"
    )

    categories_output = category_output(holdout_flows, alert_membership, promoted_ids)
    categories_output.to_csv(CATEGORY_METRICS_PATH, index=False)

    # Feb 17 is accessed only now for post-holdout diagnostics/comparison; no
    # frozen decision can be changed by this data.
    print("Building Feb 17 reference diagnostics without modifying the pipeline...", flush=True)
    validation_routing = route_validation_rows(RAW_PATH, ROUTED_VALIDATION_CSV)
    feature_manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256_file(HOLDOUT_FEATURES_PATH) != feature_manifest["outputs"]["locked_holdout"]["sha256"]:
        raise AssertionError("Holdout feature hash disagrees with frozen feature manifest")
    validation_flows, validation_alignment = restore_validation_identities(
        ROUTED_VALIDATION_CSV,
        context_features,
        baseline,
        args.memory_limit,
        args.threads,
    )
    validation_evidence, validation_membership, validation_truth = build_incident_evidence(
        validation_flows
    )
    validation_promotion_mask = (
        validation_evidence["max_attack_score"] >= PROMOTION_SCORE_THRESHOLD
    )
    validation_promoted_ids = set(
        validation_evidence.loc[validation_promotion_mask, "incident_id"]
    )
    connection = duckdb.connect()
    try:
        validation_scores_array = connection.execute(
            f"SELECT context_attack_score FROM read_parquet({sql_string(VALIDATION_SCORES_PATH)})"
        ).fetchnumpy()["context_attack_score"]
    finally:
        connection.close()

    drift = calculate_drift(
        model,
        context_features,
        np.asarray(validation_scores_array, dtype=float),
        scores,
        validation_evidence,
        evidence,
    )
    DRIFT_METRICS_PATH.write_text(
        json.dumps(json_ready(drift), indent=2), encoding="utf-8"
    )

    holdout_aggregation_patterns = pure_fp_patterns(
        holdout_flows,
        alert_membership,
        incident_truth,
        promoted_ids,
        flow_metrics["false_positives"],
        "aggregation",
    )
    holdout_promoted_patterns = pure_fp_patterns(
        holdout_flows,
        alert_membership,
        incident_truth,
        promoted_ids,
        flow_metrics["false_positives"],
        "promoted",
    )
    validation_flow_fp = int(
        ((validation_flows["label"] == 0) & (validation_flows["predicted_class"] == 1)).sum()
    )
    validation_aggregation_patterns = pure_fp_patterns(
        validation_flows,
        validation_membership,
        validation_truth,
        validation_promoted_ids,
        validation_flow_fp,
        "aggregation",
    )
    validation_promoted_patterns = pure_fp_patterns(
        validation_flows,
        validation_membership,
        validation_truth,
        validation_promoted_ids,
        validation_flow_fp,
        "promoted",
    )
    fp_patterns, fp_pattern_comparison = add_feb17_pattern_comparison(
        pd.concat(
            [holdout_aggregation_patterns, holdout_promoted_patterns],
            ignore_index=True,
        ),
        pd.concat(
            [validation_aggregation_patterns, validation_promoted_patterns],
            ignore_index=True,
        ),
    )
    fp_patterns.to_csv(FP_PATTERNS_PATH, index=False)

    stage_comparison, temporal_comparison = build_temporal_comparison(
        flow_metrics, incident_metrics
    )
    stage_comparison.to_csv(STAGE_COMPARISON_PATH, index=False)

    criteria = {
        "A_flow_recall_remains_high": flow_metrics["recall"] >= 0.985,
        "B_incident_recall_approximately_ge_99_9": before["incident_recall"]
        >= INCIDENT_RECALL_REQUIREMENT,
        "C_aggregation_materially_compresses_alerts": incident_metrics[
            "alert_to_incident_reduction_percentage"
        ]
        >= 0.50,
        "D_promotion_reduces_fp_without_breaking_incident_recall_gate": (
            after["pure_fp_promoted_incidents"] < before["pure_fp_incidents"]
            and after["incident_recall"] >= INCIDENT_RECALL_REQUIREMENT
        ),
    }
    criteria["E_no_catastrophic_temporal_degradation"] = bool(
        flow_metrics["fpr"]
        <= CATASTROPHIC_RATE_MULTIPLIER * flow_metrics["feb17_reference"]["fpr"]
        and after["fp_incidents_per_hour"]
        <= CATASTROPHIC_RATE_MULTIPLIER * feb17_after["fp_incidents_per_hour"]
    )
    if not criteria["E_no_catastrophic_temporal_degradation"]:
        verdict = "Failed temporal generalization"
    elif all(criteria.values()):
        verdict = (
            "Strong temporal generalization"
            if after["fp_incidents_per_hour"]
            <= feb17_after["fp_incidents_per_hour"]
            and after["incident_precision"] >= feb17_after["incident_precision"]
            else "Acceptable but operationally noisy"
        )
    elif (
        criteria["B_incident_recall_approximately_ge_99_9"]
        and criteria["C_aggregation_materially_compresses_alerts"]
        and criteria["D_promotion_reduces_fp_without_breaking_incident_recall_gate"]
    ):
        verdict = "Acceptable but operationally noisy"
    else:
        verdict = "Significant temporal degradation"

    hashes_after = {
        key: sha256_file(path)
        for key, path in (
            ("frozen_context_model", CONTEXT_MODEL_PATH),
            ("preprocessing_feature_manifest", FEATURE_MANIFEST_PATH),
            ("preprocessing_builder_code", FEATURE_BUILDER_PATH),
            ("threshold_configuration", ABLATION_METRICS_PATH),
            ("threshold_sweep", THRESHOLD_RESULTS_PATH),
            ("incident_aggregation_manifest", INCIDENT_AGGREGATION_MANIFEST_PATH),
            ("selected_promotion_policy", SELECTED_PROMOTION_PATH),
            ("promotion_manifest", PROMOTION_MANIFEST_PATH),
            ("source_dataset", RAW_PATH),
            ("holdout_feature_dataset", HOLDOUT_FEATURES_PATH),
        )
    }
    if hashes_after != preflight_hash_snapshot:
        raise AssertionError("A frozen input changed during final holdout evaluation")

    manifest = {
        "stage": "final temporal holdout evaluation - Feb 18",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_and_only_fully_frozen_feb18_evaluation": True,
        "preflight_integrity_before_holdout_labels": preflight,
        "frozen_hashes_unchanged_after_evaluation": hashes_after == preflight_hash_snapshot,
        "holdout_materialization": materialization,
        "holdout_scoring": scoring,
        "holdout_identity_alignment": identity_alignment,
        "model_runtime_compatibility_check": runtime_compatibility,
        "frozen_pipeline": preflight["frozen_configuration_before_holdout"],
        "data_usage_guarantees": {
            "feb18_used_for_model_training": False,
            "feb18_used_for_threshold_selection": False,
            "feb18_used_for_aggregation_selection": False,
            "feb18_used_for_promotion_selection": False,
            "all_parameters_frozen_before_feb18_access": True,
            "post_holdout_tuning_performed": False,
            "classifier_retrained": False,
            "threshold_changed": False,
            "aggregation_changed": False,
            "promotion_changed": False,
            "whitelisting_or_suppression_added": False,
            "deployment_changes_performed": False,
        },
        "flow_metrics": flow_metrics,
        "incident_metrics": incident_metrics,
        "temporal_generalization_comparison": temporal_comparison,
        "fp_pattern_comparison": fp_pattern_comparison,
        "decision_criteria": criteria,
        "predeclared_decision_thresholds": {
            "high_flow_recall_minimum": 0.985,
            "incident_recall_requirement": INCIDENT_RECALL_REQUIREMENT,
            "material_alert_compression_minimum": 0.50,
            "catastrophic_rate_multiplier_vs_feb17": CATASTROPHIC_RATE_MULTIPLIER,
        },
        "final_temporal_generalization_verdict": verdict,
        "verdict_logic_note": (
            "Criterion E uses only the predeclared 2x normalized-rate guard; "
            "the 98.5% flow-recall and 99.9% incident-recall gates remain "
            "separate criteria A and B. No score or pipeline parameter is "
            "changed by the verdict classification."
        ),
        "verdict_note": (
            "Passing the temporal holdout does not by itself make the system "
            "production-ready. Operational alert rates and drift remain relevant."
        ),
        "diagnostics_changed_pipeline": False,
        "validation_diagnostic_routing": validation_routing,
        "validation_diagnostic_alignment": validation_alignment,
        "runtime": {
            "total_seconds": time.perf_counter() - total_started,
            "holdout_routing_seconds": materialization["runtime_seconds"],
            "holdout_scoring_seconds": scoring["total_scoring_seconds"],
            "holdout_model_inference_seconds": scoring["inference_seconds"],
            "holdout_identity_alignment_seconds": identity_alignment["runtime_seconds"],
            "validation_diagnostic_alignment_seconds": validation_alignment[
                "runtime_seconds"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "model_training_scikit_learn": json.loads(
                ABLATION_METRICS_PATH.read_text(encoding="utf-8")
            )["software"]["scikit_learn"],
            "serialization_version_warning_observed": (
                sklearn.__version__
                != json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))[
                    "software"
                ]["scikit_learn"]
            ),
            "duckdb": duckdb.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {},
    }
    artifact_paths = {
        "flow_metrics": FLOW_METRICS_PATH,
        "incident_metrics": INCIDENT_METRICS_PATH,
        "category_metrics": CATEGORY_METRICS_PATH,
        "stage_comparison": STAGE_COMPARISON_PATH,
        "drift_metrics": DRIFT_METRICS_PATH,
        "fp_patterns": FP_PATTERNS_PATH,
    }
    manifest["artifacts"] = {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    }
    FINAL_MANIFEST_PATH.write_text(
        json.dumps(json_ready(manifest), indent=2), encoding="utf-8"
    )

    if not args.keep_cache:
        for path in (
            ROUTED_HOLDOUT_CSV,
            HOLDOUT_SCORES_CACHE,
            ROUTED_VALIDATION_CSV,
        ):
            path.unlink(missing_ok=True)

    print("\nFeb 18 flow metrics:")
    print(json.dumps(json_ready(flow_metrics), indent=2))
    print("\nFeb 18 incident metrics:")
    print(json.dumps(json_ready(incident_metrics), indent=2))
    print(f"\nFinal verdict: {verdict}")
    print(f"Manifest: {FINAL_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
