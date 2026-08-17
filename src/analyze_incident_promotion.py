"""Select an interpretable incident-promotion gate from Jan 22 temporal OOF data.

The Context flow detector, its 0.10 threshold, and Policy B / five-minute
aggregation are frozen.  Candidate promotion rules are selected exclusively on
causal Jan 22 OOF scores.  The selected JSON is persisted before February 17
inputs are accessed, then evaluated exactly once on that validation capture.
No classifier is loaded, fitted, tuned, calibrated, or replaced.
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
import numpy as np
import pandas as pd
import sklearn

try:
    from .analyze_incident_aggregation import (
        assign_incidents,
        restore_validation_identities,
        route_validation_rows,
        sha256_file,
    )
    from .build_temporal_features import (
        STATIC_BEHAVIORAL_FEATURES,
        baseline_features,
        context_expressions,
        read_raw_columns,
        window_clause,
    )
except ImportError:  # Direct execution: python src/analyze_incident_promotion.py
    from analyze_incident_aggregation import (
        assign_incidents,
        restore_validation_identities,
        route_validation_rows,
        sha256_file,
    )
    from build_temporal_features import (
        STATIC_BEHAVIORAL_FEATURES,
        baseline_features,
        context_expressions,
        read_raw_columns,
        window_clause,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "cic_unsw_nb15" / "CICFlowMeter_out.csv"
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
TRAIN_FEATURES_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_FEATURES_PATH = FEATURE_DIR / "validation_features.parquet"
VALIDATION_FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"

MODELS_DIR = PROJECT_ROOT / "models"
CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
OOF_SCORES_PATH = MODELS_DIR / "v2_temporal_oof_scores.parquet"
OOF_MANIFEST_PATH = MODELS_DIR / "v2_temporal_oof_manifest.json"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"

OOF_RESULTS_PATH = MODELS_DIR / "v2_incident_promotion_oof_results.csv"
SELECTED_POLICY_PATH = MODELS_DIR / "v2_incident_promotion_selected_policy.json"
VALIDATION_METRICS_PATH = MODELS_DIR / "v2_incident_promotion_validation_metrics.json"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_incident_promotion_category_metrics.csv"
MANIFEST_PATH = MODELS_DIR / "v2_incident_promotion_manifest.json"

CACHE_DIR = FEATURE_DIR / "_incident_promotion_cache"
ROUTED_TRAIN_CSV = CACHE_DIR / "jan22_routed_rows.csv"

TRAIN_CAPTURE_DATE = "2015-01-22"
TRAIN_CAPTURE_DATE_BYTES = b"22/01/2015"
EXPECTED_TRAIN_ROWS = 1_765_922
EXPECTED_OOF_ROWS = 1_701_826
EXPECTED_VALIDATION_ROWS = 498_890
FLOW_THRESHOLD = 0.10
INCIDENT_WINDOW_SECONDS = 300
INCIDENT_KEYS = ("src_ip", "dst_ip", "dst_port")
INCIDENT_POLICY = "Policy B: src_ip + dst_ip + dst_port, 5 minutes"
INCIDENT_RECALL_REQUIREMENT = 0.999
MIN_CATEGORY_REFERENCE_INCIDENTS = 20

MATCH_CONTEXT_FEATURES = (
    "dst_unique_sport_60s",
    "src_dst_conn_60s",
    "src_dport_conn_60s",
    "src_conn_60s",
    "dst_conn_60s",
    "src_packets_60s",
    "dst_packets_60s",
    "src_mean_packets_per_flow_60s",
    "dst_mean_packets_per_flow_60s",
    "src_udp_ratio_60s",
    "dst_udp_ratio_60s",
)
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

FAMILY_A_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.9001, 0.05), 2))
FAMILY_B_THRESHOLDS = (0.20, 0.30, 0.40, 0.50)
FAMILY_B_COUNTS = (3, 5, 10, 20)
FAMILY_C_THRESHOLDS = (0.30, 0.40, 0.50)
FAMILY_C_COUNTS = (2, 3, 5)


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
        help="Keep routed Jan 22 and Feb 17 CSV caches after completion.",
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
            raise ValueError("Malformed CIC row before Timestamp")
        start = comma + 1
    end = line.find(b",", start)
    if end < 0:
        raise ValueError("Malformed CIC row at Timestamp")
    return line[start:end]


def route_train_rows() -> dict[str, Any]:
    """Route only Jan 22 rows; nonmatching records remain opaque bytes."""

    started = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ROUTED_TRAIN_CSV.with_suffix(".tmp.csv")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    matched = 0
    scanned = 0
    with RAW_PATH.open("rb", buffering=8 * 1024 * 1024) as source, temporary.open(
        "wb", buffering=8 * 1024 * 1024
    ) as target:
        header = source.readline()
        if not header:
            raise ValueError("Raw CIC file is empty")
        header_columns = header.decode("utf-8-sig").rstrip("\r\n").split(",")
        if len(header_columns) != 84 or header_columns[6] != "Timestamp":
            raise AssertionError("Unexpected raw CIC schema")
        target.write(header)
        digest.update(header)
        for line in source:
            scanned += 1
            if locate_timestamp(line).startswith(TRAIN_CAPTURE_DATE_BYTES):
                target.write(line)
                digest.update(line)
                matched += 1
    if matched != EXPECTED_TRAIN_ROWS:
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"Expected {EXPECTED_TRAIN_ROWS:,} Jan 22 rows, got {matched:,}")
    os.replace(temporary, ROUTED_TRAIN_CSV)
    return {
        "rows": matched,
        "combined_source_rows_scanned": scanned,
        "materialized_non_train_rows": 0,
        "sha256": digest.hexdigest(),
        "size_bytes": ROUTED_TRAIN_CSV.stat().st_size,
        "runtime_seconds": time.perf_counter() - started,
        "routing_field": "Timestamp only",
    }


def configure_connection(
    connection: duckdb.DuckDBPyConnection, memory_limit: str, threads: int
) -> None:
    connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_string(CACHE_DIR)}")


def expression_by_alias() -> dict[str, str]:
    expressions: dict[str, str] = {}
    for expression in context_expressions():
        alias = expression.rsplit(" AS ", maxsplit=1)[-1].strip().strip('"')
        expressions[alias] = expression
    missing = sorted(set(MATCH_CONTEXT_FEATURES) - set(expressions))
    if missing:
        raise AssertionError(f"Missing context expressions: {missing}")
    return expressions


def oof_identity_query(source_relation: str, baseline: list[str]) -> str:
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


def signature_expression(columns: Iterable[str], relation_alias: str = "") -> str:
    prefix = f"{relation_alias}." if relation_alias else ""
    values = ", ".join(prefix + quote_identifier(column) for column in columns)
    return f"hash({values})"


def restore_oof_identities(
    baseline: list[str], memory_limit: str, threads: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align Jan identities to all train rows, then retain OOF attacks/alerts."""

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
            CREATE TABLE routed_train AS
            SELECT
                *,
                strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
                ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE AS flow_bytes,
                ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
            FROM read_csv_auto(
                {sql_string(ROUTED_TRAIN_CSV)},
                header = true,
                sample_size = -1,
                parallel = true
            )
            """
        )
        routed_stats = connection.execute(
            """
            SELECT count(*), min(event_ts), max(event_ts),
                   count(*) FILTER (WHERE CAST(event_ts AS DATE) != DATE '2015-01-22')
            FROM routed_train
            """
        ).fetchone()
        if routed_stats is None or tuple(map(int, (routed_stats[0], routed_stats[3]))) != (
            EXPECTED_TRAIN_ROWS,
            0,
        ):
            raise AssertionError(f"Invalid routed Jan 22 rows: {routed_stats}")

        connection.execute(
            f"CREATE TABLE identity_features AS {oof_identity_query('routed_train', baseline)}"
        )
        connection.execute(
            f"""
            CREATE TABLE frozen_train_rows AS
            SELECT row_number() OVER () - 1 AS train_row, *
            FROM read_parquet({sql_string(TRAIN_FEATURES_PATH)})
            """
        )
        connection.execute(
            f"""
            CREATE TABLE frozen_keyed AS
            SELECT *, row_number() OVER (
                PARTITION BY signature ORDER BY train_row
            ) AS duplicate_rank
            FROM (
                SELECT *, {signature} AS signature FROM frozen_train_rows
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE identity_keyed AS
            SELECT *, row_number() OVER (
                PARTITION BY signature ORDER BY timestamp, flow_id, src_ip, dst_ip
            ) AS duplicate_rank
            FROM (
                SELECT *, {signature} AS signature FROM identity_features
            )
            """
        )
        mismatch = " OR ".join(
            f"(f.{quote_identifier(column)} IS DISTINCT FROM i.{quote_identifier(column)})"
            for column in match_columns
        )
        alignment = connection.execute(
            f"""
            SELECT
                count(*) AS matched,
                count(*) FILTER (WHERE {mismatch}) AS mismatches,
                count(DISTINCT f.train_row) AS distinct_train_rows
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            """
        ).fetchone()
        if alignment is None or tuple(map(int, alignment)) != (
            EXPECTED_TRAIN_ROWS,
            0,
            EXPECTED_TRAIN_ROWS,
        ):
            raise AssertionError(f"Jan identity alignment failed: {alignment}")

        oof_stats = connection.execute(
            f"""
            SELECT
                count(*), count(DISTINCT o.train_row), min(i.timestamp), max(i.timestamp),
                sum(o.label), sum(o.context_score >= {FLOW_THRESHOLD}),
                count(*) FILTER (
                    WHERE o.label IS DISTINCT FROM f.label
                       OR o.attack_cat IS DISTINCT FROM f.attack_cat
                )
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(OOF_SCORES_PATH)}) o USING (train_row)
            """
        ).fetchone()
        if oof_stats is None or tuple(map(int, (oof_stats[0], oof_stats[1], oof_stats[6]))) != (
            EXPECTED_OOF_ROWS,
            EXPECTED_OOF_ROWS,
            0,
        ):
            raise AssertionError(f"OOF join failed: {oof_stats}")

        relevant = connection.execute(
            f"""
            SELECT
                o.train_row AS validation_row,
                o.temporal_fold,
                i.timestamp,
                i.src_ip,
                i.dst_ip,
                i."Dst Port"::BIGINT AS dst_port,
                o.attack_cat,
                o.label::UTINYINT AS label,
                o.context_score::DOUBLE AS attack_score,
                (o.context_score >= {FLOW_THRESHOLD})::UTINYINT AS predicted_class
            FROM frozen_keyed f
            JOIN identity_keyed i USING (signature, duplicate_rank)
            JOIN read_parquet({sql_string(OOF_SCORES_PATH)}) o USING (train_row)
            WHERE o.label = 1 OR o.context_score >= {FLOW_THRESHOLD}
            ORDER BY o.train_row
            """
        ).fetchdf()
    finally:
        connection.close()

    expected_relevant = int(oof_stats[4]) + int(oof_stats[5]) - int(
        ((relevant["label"] == 1) & (relevant["predicted_class"] == 1)).sum()
    )
    if len(relevant) != expected_relevant:
        raise AssertionError("Relevant OOF row union is inconsistent")
    return relevant, {
        "all_train_rows_aligned": int(alignment[0]),
        "feature_value_mismatches": int(alignment[1]),
        "oof_rows": int(oof_stats[0]),
        "oof_attack_flows": int(oof_stats[4]),
        "oof_alert_flows": int(oof_stats[5]),
        "minimum_oof_timestamp": oof_stats[2].isoformat(),
        "maximum_oof_timestamp": oof_stats[3].isoformat(),
        "materialized_relevant_rows": len(relevant),
        "match_signature_columns": len(match_columns),
        "match_context_features": list(MATCH_CONTEXT_FEATURES),
        "runtime_seconds": time.perf_counter() - started,
    }


def top3_mean(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    return float(array[-min(3, len(array)) :].mean())


def population_std(values: pd.Series) -> float:
    return float(values.to_numpy(dtype=float).std(ddof=0))


def build_incident_evidence(
    flows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Policy B incidents and label-free evidence.

    Evidence is calculated from identity, time, and frozen scores only. Labels
    are joined later into a separate evaluation table.
    """

    alerts = flows.loc[flows["predicted_class"] == 1].copy()
    evidence_input = alerts[
        ["validation_row", "timestamp", *INCIDENT_KEYS, "attack_score"]
    ].copy()
    assigned = assign_incidents(
        evidence_input, INCIDENT_KEYS, INCIDENT_WINDOW_SECONDS
    )
    grouped = assigned.groupby("incident_id", sort=False, observed=True)
    evidence = grouped.agg(
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        alert_flow_count=("validation_row", "size"),
        max_attack_score=("attack_score", "max"),
        mean_attack_score=("attack_score", "mean"),
        median_attack_score=("attack_score", "median"),
        p90_attack_score=("attack_score", lambda values: float(values.quantile(0.90))),
        top3_mean_attack_score=("attack_score", top3_mean),
        flows_score_ge_0_20=("attack_score", lambda values: int((values >= 0.20).sum())),
        flows_score_ge_0_30=("attack_score", lambda values: int((values >= 0.30).sum())),
        flows_score_ge_0_40=("attack_score", lambda values: int((values >= 0.40).sum())),
        flows_score_ge_0_50=("attack_score", lambda values: int((values >= 0.50).sum())),
        score_std=("attack_score", population_std),
    ).reset_index()
    evidence["duration_seconds"] = (
        evidence["last_timestamp"] - evidence["first_timestamp"]
    ).dt.total_seconds()
    keys = grouped[list(INCIDENT_KEYS)].first().reset_index()
    evidence = evidence.merge(keys, on="incident_id", validate="1:1")

    evidence_columns = {
        "incident_id",
        "first_timestamp",
        "last_timestamp",
        *INCIDENT_KEYS,
        "alert_flow_count",
        "duration_seconds",
        "max_attack_score",
        "mean_attack_score",
        "median_attack_score",
        "p90_attack_score",
        "top3_mean_attack_score",
        "flows_score_ge_0_20",
        "flows_score_ge_0_30",
        "flows_score_ge_0_40",
        "flows_score_ge_0_50",
        "score_std",
    }
    if set(evidence.columns) != evidence_columns:
        raise AssertionError("Incident evidence schema changed")
    if {"label", "attack_cat"} & set(evidence.columns):
        raise AssertionError("Ground truth leaked into incident evidence")

    membership = assigned[["validation_row", "incident_id"]].copy()
    truth = membership.merge(
        alerts[["validation_row", "label"]], on="validation_row", validate="1:1"
    )
    incident_truth = (
        truth.groupby("incident_id", observed=True)["label"]
        .agg(attack_alert_flows="sum", member_alert_flows="size")
        .reset_index()
    )
    incident_truth["normal_alert_flows"] = (
        incident_truth["member_alert_flows"] - incident_truth["attack_alert_flows"]
    )
    incident_truth["tp_or_mixed"] = incident_truth["attack_alert_flows"] > 0
    incident_truth["pure_false_positive"] = incident_truth["attack_alert_flows"] == 0
    incident_truth["mixed"] = (incident_truth["attack_alert_flows"] > 0) & (
        incident_truth["normal_alert_flows"] > 0
    )
    return evidence, membership, incident_truth


def build_reference_attack_incidents(
    flows: pd.DataFrame,
    category: str = "ALL_ATTACKS",
) -> pd.DataFrame:
    attacks = flows.loc[flows["label"] == 1].copy()
    if category != "ALL_ATTACKS":
        attacks = attacks.loc[attacks["attack_cat"] == category].copy()
    assigned = assign_incidents(attacks, INCIDENT_KEYS, INCIDENT_WINDOW_SECONDS)
    assigned = assigned.rename(columns={"incident_id": "reference_incident_id"})
    return assigned[["validation_row", "reference_incident_id"]]


def candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in FAMILY_A_THRESHOLDS:
        candidates.append(
            {
                "family": "A",
                "score_threshold": float(threshold),
                "count_threshold": None,
                "count_field": None,
                "complexity_rank": 1,
                "rule": f"max_attack_score >= {threshold:.2f}",
            }
        )
    for threshold in FAMILY_B_THRESHOLDS:
        for count in FAMILY_B_COUNTS:
            candidates.append(
                {
                    "family": "B",
                    "score_threshold": threshold,
                    "count_threshold": count,
                    "count_field": "alert_flow_count",
                    "complexity_rank": 2,
                    "rule": (
                        f"max_attack_score >= {threshold:.2f} OR "
                        f"alert_flow_count >= {count}"
                    ),
                }
            )
    for threshold in FAMILY_C_THRESHOLDS:
        for count in FAMILY_C_COUNTS:
            candidates.append(
                {
                    "family": "C",
                    "score_threshold": threshold,
                    "count_threshold": count,
                    "count_field": "flows_score_ge_0_30",
                    "complexity_rank": 3,
                    "rule": (
                        f"max_attack_score >= {threshold:.2f} OR "
                        f"flows_score_ge_0_30 >= {count}"
                    ),
                }
            )
    if len(candidates) != 42:
        raise AssertionError("Promotion candidate grid must contain exactly 42 rules")
    return candidates


def promoted_mask(evidence: pd.DataFrame, candidate: dict[str, Any]) -> pd.Series:
    mask = evidence["max_attack_score"] >= candidate["score_threshold"]
    if candidate["family"] in {"B", "C"}:
        mask = mask | (
            evidence[str(candidate["count_field"])] >= int(candidate["count_threshold"])
        )
    return mask


def reference_detection_count(
    reference_membership: pd.DataFrame,
    alert_membership: pd.DataFrame,
    promoted_incident_ids: set[str],
) -> int:
    joined = reference_membership.merge(
        alert_membership, on="validation_row", how="left", validate="1:1"
    )
    joined["row_promoted"] = joined["incident_id"].isin(promoted_incident_ids)
    return int(
        joined.groupby("reference_incident_id", observed=True)["row_promoted"].max().sum()
    )


def evaluate_candidate(
    evidence: pd.DataFrame,
    alert_membership: pd.DataFrame,
    incident_truth: pd.DataFrame,
    reference_membership: pd.DataFrame,
    candidate: dict[str, Any],
    capture_hours: float,
) -> dict[str, Any]:
    mask = promoted_mask(evidence, candidate)
    promoted_ids = set(evidence.loc[mask, "incident_id"])
    promoted_truth = incident_truth.loc[
        incident_truth["incident_id"].isin(promoted_ids)
    ]
    total_reference = int(reference_membership["reference_incident_id"].nunique())
    promoted_reference = reference_detection_count(
        reference_membership, alert_membership, promoted_ids
    )
    promoted_total = len(promoted_truth)
    promoted_tp = int(promoted_truth["tp_or_mixed"].sum())
    promoted_fp = int(promoted_truth["pure_false_positive"].sum())
    ungated_total = len(evidence)
    row = {
        **candidate,
        "reference_attack_incidents": total_reference,
        "detected_promoted_attack_incidents": promoted_reference,
        "missed_attack_incidents": total_reference - promoted_reference,
        "incident_recall": promoted_reference / total_reference,
        "promoted_total_incidents": promoted_total,
        "promoted_tp_or_mixed_incidents": promoted_tp,
        "promoted_pure_fp_incidents": promoted_fp,
        "incident_precision": promoted_tp / promoted_total if promoted_total else np.nan,
        "fp_incidents_per_hour": promoted_fp / capture_hours,
        "total_promoted_incidents_per_hour": promoted_total / capture_hours,
        "ungated_total_incidents": ungated_total,
        "absolute_incident_reduction_vs_ungated": ungated_total - promoted_total,
        "incident_reduction_percentage_vs_ungated": 1.0 - promoted_total / ungated_total,
        "recall_requirement_achieved": (
            promoted_reference / total_reference >= INCIDENT_RECALL_REQUIREMENT
        ),
        "selected": False,
    }
    if promoted_total != promoted_tp + promoted_fp:
        raise AssertionError("Promoted incident labels do not partition incidents")
    return row


def evaluate_all_candidates(
    evidence: pd.DataFrame,
    alert_membership: pd.DataFrame,
    incident_truth: pd.DataFrame,
    reference_membership: pd.DataFrame,
    capture_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    results = pd.DataFrame(
        [
            evaluate_candidate(
                evidence,
                alert_membership,
                incident_truth,
                reference_membership,
                candidate,
                capture_hours,
            )
            for candidate in candidate_grid()
        ]
    )
    eligible = results.loc[results["recall_requirement_achieved"]].copy()
    gate_achieved = not eligible.empty
    if gate_achieved:
        ordered = eligible.sort_values(
            [
                "promoted_pure_fp_incidents",
                "promoted_total_incidents",
                "complexity_rank",
                "incident_precision",
            ],
            ascending=[True, True, True, False],
        )
    else:
        ordered = results.sort_values(
            [
                "incident_recall",
                "promoted_pure_fp_incidents",
                "promoted_total_incidents",
                "complexity_rank",
                "incident_precision",
            ],
            ascending=[False, True, True, True, False],
        )
    selected_index = ordered.index[0]
    results.loc[selected_index, "selected"] = True
    selected = results.loc[selected_index].to_dict()
    return results, selected, gate_achieved


def category_metrics(
    split: str,
    flows: pd.DataFrame,
    alert_membership: pd.DataFrame,
    promoted_incident_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in ("ALL_ATTACKS", *ATTACK_CATEGORIES):
        category_flows = flows.loc[flows["label"] == 1]
        if category != "ALL_ATTACKS":
            category_flows = category_flows.loc[
                category_flows["attack_cat"] == category
            ]
        reference = build_reference_attack_incidents(flows, category)
        total_incidents = int(reference["reference_incident_id"].nunique())
        promoted_incidents = reference_detection_count(
            reference, alert_membership, promoted_incident_ids
        )
        alert_ids = set(alert_membership["incident_id"])
        ungated_detected = reference_detection_count(
            reference, alert_membership, alert_ids
        )
        total_flows = len(category_flows)
        detected_flows = int(category_flows["predicted_class"].sum())
        rows.append(
            {
                "split": split,
                "attack_category": category,
                "attack_flows": total_flows,
                "detected_attack_flows_at_flow_threshold": detected_flows,
                "flow_recall": detected_flows / total_flows,
                "reference_attack_incidents": total_incidents,
                "ungated_detected_attack_incidents": ungated_detected,
                "ungated_incident_recall": ungated_detected / total_incidents,
                "promoted_detected_attack_incidents": promoted_incidents,
                "promoted_missed_attack_incidents": total_incidents - promoted_incidents,
                "promoted_incident_recall": promoted_incidents / total_incidents,
                "sufficient_ground_truth": (
                    total_incidents >= MIN_CATEGORY_REFERENCE_INCIDENTS
                ),
                "sufficiency_rule": (
                    f"reference_attack_incidents >= {MIN_CATEGORY_REFERENCE_INCIDENTS}"
                ),
            }
        )
    return pd.DataFrame(rows)


def ungated_metrics(
    evidence: pd.DataFrame,
    alert_membership: pd.DataFrame,
    incident_truth: pd.DataFrame,
    reference_membership: pd.DataFrame,
    capture_hours: float,
) -> dict[str, Any]:
    all_ids = set(evidence["incident_id"])
    total_reference = int(reference_membership["reference_incident_id"].nunique())
    detected_reference = reference_detection_count(
        reference_membership, alert_membership, all_ids
    )
    fp = int(incident_truth["pure_false_positive"].sum())
    tp = int(incident_truth["tp_or_mixed"].sum())
    return {
        "total_incidents": len(evidence),
        "pure_fp_incidents": fp,
        "tp_or_mixed_incidents": tp,
        "mixed_incidents": int(incident_truth["mixed"].sum()),
        "reference_attack_incidents": total_reference,
        "detected_attack_incidents": detected_reference,
        "missed_attack_incidents": total_reference - detected_reference,
        "incident_recall": detected_reference / total_reference,
        "incident_precision": tp / len(evidence),
        "incidents_per_hour": len(evidence) / capture_hours,
        "fp_incidents_per_hour": fp / capture_hours,
    }


def promoted_validation_metrics(
    evidence: pd.DataFrame,
    alert_membership: pd.DataFrame,
    incident_truth: pd.DataFrame,
    reference_membership: pd.DataFrame,
    selected_rule: dict[str, Any],
    capture_hours: float,
) -> tuple[dict[str, Any], set[str]]:
    before = ungated_metrics(
        evidence,
        alert_membership,
        incident_truth,
        reference_membership,
        capture_hours,
    )
    mask = promoted_mask(evidence, selected_rule)
    promoted_ids = set(evidence.loc[mask, "incident_id"])
    promoted_truth = incident_truth.loc[
        incident_truth["incident_id"].isin(promoted_ids)
    ]
    promoted_reference = reference_detection_count(
        reference_membership, alert_membership, promoted_ids
    )
    total_reference = int(reference_membership["reference_incident_id"].nunique())
    promoted_total = len(promoted_truth)
    promoted_tp = int(promoted_truth["tp_or_mixed"].sum())
    promoted_fp = int(promoted_truth["pure_false_positive"].sum())
    after = {
        "total_promoted_incidents": promoted_total,
        "pure_fp_promoted_incidents": promoted_fp,
        "tp_or_mixed_promoted_incidents": promoted_tp,
        "mixed_promoted_incidents": int(promoted_truth["mixed"].sum()),
        "reference_attack_incidents": total_reference,
        "promoted_detected_attack_incidents": promoted_reference,
        "promoted_missed_attack_incidents": total_reference - promoted_reference,
        "incident_recall": promoted_reference / total_reference,
        "incident_precision": promoted_tp / promoted_total,
        "incidents_per_hour": promoted_total / capture_hours,
        "fp_incidents_per_hour": promoted_fp / capture_hours,
    }
    reductions = {
        "total_incidents_absolute": before["total_incidents"] - promoted_total,
        "total_incidents_percentage": 1.0 - promoted_total / before["total_incidents"],
        "pure_fp_incidents_absolute": before["pure_fp_incidents"] - promoted_fp,
        "pure_fp_incidents_percentage": 1.0 - promoted_fp / before["pure_fp_incidents"],
        "incidents_per_hour_absolute": before["incidents_per_hour"]
        - after["incidents_per_hour"],
        "fp_incidents_per_hour_absolute": before["fp_incidents_per_hour"]
        - after["fp_incidents_per_hour"],
    }
    return {"before_promotion": before, "after_promotion": after, "reductions": reductions}, promoted_ids


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    total_started = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        RAW_PATH,
        TRAIN_FEATURES_PATH,
        OOF_SCORES_PATH,
        OOF_MANIFEST_PATH,
        CONTEXT_MODEL_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Selection phase: Jan 22 and its temporal OOF artifacts only.
    oof_manifest = json.loads(OOF_MANIFEST_PATH.read_text(encoding="utf-8"))
    if oof_manifest["source_split"] != "2015-01-22 only":
        raise AssertionError("Temporal OOF manifest is not Jan 22-only")
    if not oof_manifest["every_saved_oof_row_scored_once"]:
        raise AssertionError("Temporal OOF scores are incomplete")
    if oof_manifest["future_rows_used_for_base_model_training"]:
        raise AssertionError("Temporal OOF manifest reports future-row leakage")

    baseline = baseline_features(read_raw_columns(RAW_PATH))
    model_hash_before = sha256_file(CONTEXT_MODEL_PATH)
    oof_score_hash = sha256_file(OOF_SCORES_PATH)
    train_hash = sha256_file(TRAIN_FEATURES_PATH)
    raw_hash = sha256_file(RAW_PATH)
    if oof_score_hash != oof_manifest["oof_scores_sha256"]:
        raise AssertionError("Temporal OOF score artifact hash changed")
    if train_hash != oof_manifest["train_parquet_sha256"]:
        raise AssertionError("Jan 22 feature dataset hash changed")

    print("Routing Jan 22 rows for OOF identity reconstruction...", flush=True)
    train_routing = route_train_rows()
    print("Restoring Jan 22 OOF timestamps and Policy B entities...", flush=True)
    oof_flows, oof_alignment = restore_oof_identities(
        baseline, args.memory_limit, args.threads
    )
    oof_hours = (
        (
            pd.Timestamp(oof_alignment["maximum_oof_timestamp"])
            - pd.Timestamp(oof_alignment["minimum_oof_timestamp"])
        ).total_seconds()
        + 1.0
    ) / 3600.0
    oof_evidence, oof_membership, oof_truth = build_incident_evidence(oof_flows)
    oof_reference = build_reference_attack_incidents(oof_flows)
    oof_before = ungated_metrics(
        oof_evidence, oof_membership, oof_truth, oof_reference, oof_hours
    )
    oof_results, selected, gate_achieved = evaluate_all_candidates(
        oof_evidence,
        oof_membership,
        oof_truth,
        oof_reference,
        oof_hours,
    )
    selected_rule = {
        key: (None if pd.isna(selected[key]) else selected[key])
        for key in (
            "family",
            "score_threshold",
            "count_threshold",
            "count_field",
            "complexity_rank",
            "rule",
        )
    }
    selected_mask = promoted_mask(oof_evidence, selected_rule)
    selected_oof_ids = set(oof_evidence.loc[selected_mask, "incident_id"])
    oof_categories = category_metrics(
        "selection_oof", oof_flows, oof_membership, selected_oof_ids
    )

    oof_results = oof_results.sort_values(
        ["family", "score_threshold", "count_threshold"], na_position="first"
    ).reset_index(drop=True)
    oof_results.to_csv(OOF_RESULTS_PATH, index=False)
    freeze_time = datetime.now(timezone.utc).isoformat()
    selected_policy_document = {
        "stage": "incident promotion rule frozen from Jan 22 temporal OOF",
        "frozen_at_utc": freeze_time,
        "selection_data": "Jan 22 causal temporal OOF only",
        "flow_threshold": FLOW_THRESHOLD,
        "aggregation_policy": INCIDENT_POLICY,
        "selection_requirement": (
            "incident-level attack recall >= 0.999, then minimum pure FP "
            "incidents; tie-break by lower total incidents, simpler rule, "
            "higher precision"
        ),
        "recall_requirement_achieved": gate_achieved,
        "selected_rule": selected_rule,
        "selected_oof_metrics": {
            key: selected[key]
            for key in (
                "reference_attack_incidents",
                "detected_promoted_attack_incidents",
                "missed_attack_incidents",
                "incident_recall",
                "promoted_total_incidents",
                "promoted_tp_or_mixed_incidents",
                "promoted_pure_fp_incidents",
                "incident_precision",
                "fp_incidents_per_hour",
                "total_promoted_incidents_per_hour",
                "absolute_incident_reduction_vs_ungated",
                "incident_reduction_percentage_vs_ungated",
            )
        },
        "ungated_oof_metrics": oof_before,
        "candidate_grid": {
            "family_a": {
                "thresholds": list(FAMILY_A_THRESHOLDS),
                "rule": "max_attack_score >= T",
            },
            "family_b": {
                "thresholds": list(FAMILY_B_THRESHOLDS),
                "counts": list(FAMILY_B_COUNTS),
                "rule": "max_attack_score >= T OR alert_flow_count >= N",
            },
            "family_c": {
                "thresholds": list(FAMILY_C_THRESHOLDS),
                "counts": list(FAMILY_C_COUNTS),
                "rule": (
                    "max_attack_score >= T OR flows_score_ge_0_30 >= N"
                ),
            },
            "candidate_count": 42,
        },
        "selection_input_hashes": {
            "frozen_context_model_sha256": model_hash_before,
            "temporal_oof_scores_sha256": oof_score_hash,
            "jan22_features_sha256": train_hash,
            "combined_raw_source_sha256": raw_hash,
        },
        "oof_results_path": str(OOF_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "oof_results_sha256": sha256_file(OOF_RESULTS_PATH),
        "validation_accessed_before_freeze": False,
    }
    SELECTED_POLICY_PATH.write_text(
        json.dumps(json_ready(selected_policy_document), indent=2), encoding="utf-8"
    )
    selected_policy_hash = sha256_file(SELECTED_POLICY_PATH)
    print(
        f"Frozen OOF-selected rule before validation access: {selected_rule['rule']}",
        flush=True,
    )

    # Validation phase begins only after the selected policy file exists.
    for path in (
        VALIDATION_FEATURES_PATH,
        VALIDATION_FEATURE_MANIFEST_PATH,
        VALIDATION_SCORES_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation_feature_hash = sha256_file(VALIDATION_FEATURES_PATH)
    validation_score_hash = sha256_file(VALIDATION_SCORES_PATH)
    feature_manifest_hash = sha256_file(VALIDATION_FEATURE_MANIFEST_PATH)
    feature_manifest = json.loads(
        VALIDATION_FEATURE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    context_features = [
        *baseline,
        *feature_manifest["feature_contract"]["context_v2_static_behavioral_features"],
        *feature_manifest["feature_contract"]["context_v2_temporal_features"],
    ]
    if len(context_features) != 128:
        raise AssertionError("Validation feature contract changed")
    if validation_feature_hash != feature_manifest["outputs"]["validation"]["sha256"]:
        raise AssertionError("Feb 17 feature hash changed")

    print("Routing and loading Feb 17 once for frozen-rule validation...", flush=True)
    validation_routing = route_validation_rows(RAW_PATH, CACHE_DIR / "feb17_routed_rows.csv")
    validation_flows, validation_alignment = restore_validation_identities(
        CACHE_DIR / "feb17_routed_rows.csv",
        context_features,
        baseline,
        args.memory_limit,
        args.threads,
    )
    if len(validation_flows) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError("Feb 17 validation row count changed")
    validation_hours = (
        (validation_flows["timestamp"].max() - validation_flows["timestamp"].min()).total_seconds()
        + 1.0
    ) / 3600.0
    validation_evidence, validation_membership, validation_truth = build_incident_evidence(
        validation_flows
    )
    validation_reference = build_reference_attack_incidents(validation_flows)
    validation_metrics, validation_promoted_ids = promoted_validation_metrics(
        validation_evidence,
        validation_membership,
        validation_truth,
        validation_reference,
        selected_rule,
        validation_hours,
    )
    validation_categories = category_metrics(
        "validation_feb17",
        validation_flows,
        validation_membership,
        validation_promoted_ids,
    )
    category_output = pd.concat(
        [oof_categories, validation_categories], ignore_index=True
    )
    category_output.to_csv(CATEGORY_METRICS_PATH, index=False)

    validation_document = {
        "stage": "one-time Feb 17 validation of frozen incident promotion rule",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_policy_sha256_before_validation": selected_policy_hash,
        "selected_rule": selected_rule,
        "flow_threshold": FLOW_THRESHOLD,
        "aggregation_policy": INCIDENT_POLICY,
        "capture_hours_inclusive": validation_hours,
        **validation_metrics,
        "category_metrics_path": str(CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)),
        "rule_modified_after_validation": False,
    }
    VALIDATION_METRICS_PATH.write_text(
        json.dumps(json_ready(validation_document), indent=2), encoding="utf-8"
    )
    if sha256_file(SELECTED_POLICY_PATH) != selected_policy_hash:
        raise AssertionError("Selected promotion policy changed after validation")

    model_hash_after = sha256_file(CONTEXT_MODEL_PATH)
    if model_hash_after != model_hash_before:
        raise AssertionError("Frozen Context model changed")
    manifest = {
        "stage": "v2 deterministic incident-level promotion analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "classifier_retrained": False,
            "classifier_loaded": False,
            "classifier_modified": False,
            "flow_threshold_changed": False,
            "aggregation_policy_changed": False,
            "new_classifier_trained": False,
            "entity_whitelisting_used": False,
            "port_suppression_used": False,
            "validation_used_for_rule_selection": False,
            "validation_evaluations": 1,
            "locked_holdout_loaded": False,
            "locked_holdout_scored": False,
            "locked_holdout_evaluated": False,
        },
        "frozen_components": {
            "context_model_sha256": model_hash_before,
            "flow_threshold": FLOW_THRESHOLD,
            "aggregation_policy": "B",
            "grouping_key": list(INCIDENT_KEYS),
            "temporal_window_seconds": INCIDENT_WINDOW_SECONDS,
        },
        "frozen_score_hashes": {
            "jan22_temporal_oof_scores": oof_score_hash,
            "feb17_validation_scores": validation_score_hash,
        },
        "dataset_hashes": {
            "jan22_features": train_hash,
            "feb17_features": validation_feature_hash,
            "feature_manifest": feature_manifest_hash,
            "combined_raw_source": raw_hash,
            "routed_jan22_rows": train_routing["sha256"],
            "routed_feb17_rows": validation_routing["sha256"],
        },
        "candidate_policy_grid": selected_policy_document["candidate_grid"],
        "selection_requirement": selected_policy_document["selection_requirement"],
        "selected_rule": selected_rule,
        "selected_policy_sha256": selected_policy_hash,
        "selected_policy_frozen_before_validation": True,
        "temporal_oof_construction": {
            key: oof_manifest[key]
            for key in (
                "source_split",
                "row_order",
                "timestamp_column_used_as_model_feature",
                "block_boundaries",
                "warmup_rows",
                "folds",
                "boundary_purge_rows_per_fold",
                "every_saved_oof_row_scored_once",
                "future_rows_used_for_base_model_training",
            )
        },
        "incident_semantics": {
            "sort": "per key by Timestamp ascending, then frozen row id",
            "new_incident": "gap from preceding alert for the same key > 300 seconds",
            "same_second": "zero gap; same incident for the same key",
            "reference_incidents": (
                "actual attack flows grouped independently with the same Policy B / 5m"
            ),
            "promotion_online_equivalence": (
                "All candidate rules are monotonic in max/count evidence; final "
                "promotion status equals whether the rule would ever trigger while "
                "the incident accumulates chronologically."
            ),
        },
        "incident_evidence_fields": [
            "alert_flow_count",
            "duration_seconds",
            "max_attack_score",
            "mean_attack_score",
            "median_attack_score",
            "p90_attack_score",
            "top3_mean_attack_score",
            "flows_score_ge_0_20",
            "flows_score_ge_0_30",
            "flows_score_ge_0_40",
            "flows_score_ge_0_50",
            "score_std",
        ],
        "ground_truth_excluded_from_evidence": True,
        "oof_identity_routing": train_routing,
        "oof_identity_alignment": oof_alignment,
        "validation_identity_routing": validation_routing,
        "validation_identity_alignment": validation_alignment,
        "oof_ungated_metrics": oof_before,
        "oof_selected_metrics": selected_policy_document["selected_oof_metrics"],
        "validation_metrics": validation_metrics,
        "new_operational_pareto_improvement": bool(
            validation_metrics["after_promotion"]["incident_recall"]
            >= INCIDENT_RECALL_REQUIREMENT
            and validation_metrics["after_promotion"]["pure_fp_promoted_incidents"]
            < validation_metrics["before_promotion"]["pure_fp_incidents"]
            and validation_metrics["after_promotion"]["total_promoted_incidents"]
            < validation_metrics["before_promotion"]["total_incidents"]
        ),
        "runtime": {
            "total_seconds": time.perf_counter() - total_started,
            "jan22_routing_seconds": train_routing["runtime_seconds"],
            "oof_identity_alignment_seconds": oof_alignment["runtime_seconds"],
            "feb17_routing_seconds": validation_routing["runtime_seconds"],
            "feb17_identity_alignment_seconds": validation_alignment["runtime_seconds"],
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
            "oof_results": str(OOF_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "oof_results_sha256": sha256_file(OOF_RESULTS_PATH),
            "selected_policy": str(SELECTED_POLICY_PATH.relative_to(PROJECT_ROOT)),
            "selected_policy_sha256": selected_policy_hash,
            "validation_metrics": str(VALIDATION_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "validation_metrics_sha256": sha256_file(VALIDATION_METRICS_PATH),
            "category_metrics": str(CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "category_metrics_sha256": sha256_file(CATEGORY_METRICS_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(json_ready(manifest), indent=2), encoding="utf-8"
    )

    if not args.keep_cache:
        ROUTED_TRAIN_CSV.unlink(missing_ok=True)
        (CACHE_DIR / "feb17_routed_rows.csv").unlink(missing_ok=True)

    print("\nSelected OOF rule:")
    print(json.dumps(json_ready(selected_policy_document["selected_oof_metrics"]), indent=2))
    print("\nFeb 17 validation:")
    print(json.dumps(json_ready(validation_metrics), indent=2))
    print(f"\nOOF results: {OOF_RESULTS_PATH}")
    print(f"Selected policy: {SELECTED_POLICY_PATH}")
    print(f"Validation metrics: {VALIDATION_METRICS_PATH}")
    print(f"Category metrics: {CATEGORY_METRICS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
