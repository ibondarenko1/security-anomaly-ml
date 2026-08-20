"""Verify the complete frozen product detector on raw unlabeled Feb 17 flows.

The product path receives no ground truth. Research helpers and frozen score
artifacts are used only after ``SecurityAnomalyDetector.analyze_batch`` returns
to prove flow, Policy B / five-minute incident, and promotion parity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_incident_aggregation import assign_incidents  # noqa: E402
from src.security_anomaly.detector import SecurityAnomalyDetector  # noqa: E402
from src.security_anomaly.results import IncidentDetection  # noqa: E402
from tools.verify_product_pipeline_parity import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_FROZEN_FEATURES,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_RAW_FEB17,
    DEFAULT_REFERENCE_SCORES,
    DEFAULT_TEMP_DIRECTORY,
    EXPECTED_FROZEN_FEATURES_SHA256,
    EXPECTED_RAW_FEB17_SHA256,
    EXPECTED_REFERENCE_SCORES_SHA256,
    EXPECTED_ROWS,
    PARITY_SOURCE_ROW,
    build_lossless_alignment,
    configure_connection,
    load_feb17_only_csv,
    remove_evaluation_columns,
    verify_file,
)


FLOW_THRESHOLD = 0.10
INCIDENT_KEYS = ("src_ip", "dst_ip", "dst_port")
INCIDENT_WINDOW_SECONDS = 300
PROMOTION_THRESHOLD = 0.25
EXPECTED_FLOW_ALERTS = 32_164
EXPECTED_INCIDENTS = 5_408
EXPECTED_PROMOTED_INCIDENTS = 5_274


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-feb17", type=Path, default=DEFAULT_RAW_FEB17)
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
        "--report-path",
        type=Path,
        help="Optional local JSON report; dataset-derived reports are not committed.",
    )
    return parser.parse_args()


def reference_incident_table(reference_flows: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen research implementation to reference-score alerts."""

    alerts = reference_flows.loc[reference_flows["is_alert"]].copy()
    assigned = assign_incidents(alerts, INCIDENT_KEYS, INCIDENT_WINDOW_SECONDS)
    grouped = assigned.groupby("incident_id", sort=False, observed=True)
    incidents = grouped.agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        src_ip=("src_ip", "first"),
        dst_ip=("dst_ip", "first"),
        dst_port=("dst_port", "first"),
        flow_count=("validation_row", "size"),
        max_attack_score=("attack_score", "max"),
        mean_attack_score=("attack_score", "mean"),
        validation_rows=("validation_row", lambda values: tuple(sorted(values))),
    ).reset_index(drop=True)
    incidents["promoted"] = incidents["max_attack_score"] >= PROMOTION_THRESHOLD
    return incidents


def product_incident_table(
    incidents: tuple[IncidentDetection, ...], source_to_validation: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for incident in incidents:
        validation_rows = tuple(
            sorted(
                int(source_to_validation[source_row])
                for source_row in incident.source_row_ids
            )
        )
        rows.append(
            {
                "incident_sequence": incident.incident_sequence,
                "first_seen": pd.Timestamp(incident.first_seen),
                "last_seen": pd.Timestamp(incident.last_seen),
                "src_ip": incident.src_ip,
                "dst_ip": incident.dst_ip,
                "dst_port": incident.dst_port,
                "flow_count": incident.flow_count,
                "max_attack_score": incident.max_attack_score,
                "mean_attack_score": incident.mean_attack_score,
                "promoted": incident.promoted,
                "validation_rows": validation_rows,
            }
        )
    return pd.DataFrame(rows)


def compare_incident_tables(
    product: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, Any]:
    """Compare canonical membership first, then every operational attribute."""

    product_by_members = {
        row.validation_rows: row for row in product.itertuples(index=False)
    }
    reference_by_members = {
        row.validation_rows: row for row in reference.itertuples(index=False)
    }
    product_memberships = set(product_by_members)
    reference_memberships = set(reference_by_members)
    missing_memberships = reference_memberships - product_memberships
    extra_memberships = product_memberships - reference_memberships
    key_disagreements = 0
    timestamp_disagreements = 0
    flow_count_disagreements = 0
    max_score_disagreements = 0
    mean_score_disagreements = 0
    promotion_disagreements = 0
    maximum_max_score_difference = 0.0
    maximum_mean_score_difference = 0.0
    for membership in product_memberships & reference_memberships:
        observed = product_by_members[membership]
        expected = reference_by_members[membership]
        if (observed.src_ip, observed.dst_ip, int(observed.dst_port)) != (
            expected.src_ip,
            expected.dst_ip,
            int(expected.dst_port),
        ):
            key_disagreements += 1
        if (
            pd.Timestamp(observed.first_seen) != pd.Timestamp(expected.first_seen)
            or pd.Timestamp(observed.last_seen) != pd.Timestamp(expected.last_seen)
        ):
            timestamp_disagreements += 1
        if int(observed.flow_count) != int(expected.flow_count):
            flow_count_disagreements += 1
        max_difference = abs(
            float(observed.max_attack_score) - float(expected.max_attack_score)
        )
        mean_difference = abs(
            float(observed.mean_attack_score) - float(expected.mean_attack_score)
        )
        maximum_max_score_difference = max(maximum_max_score_difference, max_difference)
        maximum_mean_score_difference = max(
            maximum_mean_score_difference, mean_difference
        )
        if max_difference > 1e-12:
            max_score_disagreements += 1
        if mean_difference > 1e-12:
            mean_score_disagreements += 1
        if bool(observed.promoted) != bool(expected.promoted):
            promotion_disagreements += 1
    return {
        "product_incidents": len(product),
        "reference_incidents": len(reference),
        "missing_reference_memberships": len(missing_memberships),
        "extra_product_memberships": len(extra_memberships),
        "exact_membership_parity": not missing_memberships and not extra_memberships,
        "grouping_key_disagreements": key_disagreements,
        "first_last_timestamp_disagreements": timestamp_disagreements,
        "flow_count_disagreements": flow_count_disagreements,
        "max_score_disagreements_gt_1e_12": max_score_disagreements,
        "mean_score_disagreements_gt_1e_12": mean_score_disagreements,
        "maximum_max_score_difference": maximum_max_score_difference,
        "maximum_mean_score_difference": maximum_mean_score_difference,
        "promotion_decision_disagreements": promotion_disagreements,
    }


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

    detector = SecurityAnomalyDetector.load(
        model_path=args.model,
        manifest_path=args.manifest,
        feature_contract_path=args.contract,
    )
    print("Loading isolated raw Feb 17 flows...", flush=True)
    raw = load_feb17_only_csv(args.raw_feb17)
    raw, removed_evaluation_columns = remove_evaluation_columns(
        raw, detector.bundle.contract
    )
    product_input_columns = tuple(raw.columns)
    identity_source = pd.DataFrame(
        {
            "source_row_id": np.arange(EXPECTED_ROWS, dtype=np.int64),
            "timestamp": pd.to_datetime(
                raw["Timestamp"],
                format="%d/%m/%Y %I:%M:%S %p",
                errors="raise",
            ),
            "src_ip": raw["Src IP"].astype(str),
            "dst_ip": raw["Dst IP"].astype(str),
            "dst_port": pd.to_numeric(raw["Dst Port"]).astype(np.int64),
        }
    )
    print("Running complete label-free product detector...", flush=True)
    product = detector.analyze_batch(raw)
    if product.flows_processed != EXPECTED_ROWS:
        raise AssertionError(f"Product processed {product.flows_processed:,} flows")

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
        print("Proving stable source-to-frozen row alignment...", flush=True)
        validation_to_source, alignment_report = build_lossless_alignment(
            connection,
            frozen_features_path=args.frozen_features,
            contract=detector.bundle.contract,
        )
        connection.unregister("alignment_input")
        del raw
        gc.collect()
        for table in (
            "research_generated",
            "research_keyed",
            "frozen_keyed",
            "frozen_rows",
        ):
            connection.execute(f"DROP TABLE {table}")
        reference_scores = connection.execute(
            f"""
            SELECT context_attack_score::DOUBLE AS reference_attack_score
            FROM read_parquet({sql_string(args.reference_scores)})
            ORDER BY validation_row
            """
        ).fetchnumpy()["reference_attack_score"]
    finally:
        connection.close()

    source_to_validation = np.empty(EXPECTED_ROWS, dtype=np.int64)
    source_to_validation[validation_to_source] = np.arange(
        EXPECTED_ROWS, dtype=np.int64
    )
    product_scores_by_source = np.empty(EXPECTED_ROWS, dtype=float)
    product_alerts_by_source = np.empty(EXPECTED_ROWS, dtype=bool)
    observed_source_rows = np.zeros(EXPECTED_ROWS, dtype=bool)
    for detection in product.flow_detections:
        source_row = detection.source_row_id
        if observed_source_rows[source_row]:
            raise AssertionError(f"Duplicate product source row: {source_row}")
        observed_source_rows[source_row] = True
        product_scores_by_source[source_row] = detection.attack_score
        product_alerts_by_source[source_row] = detection.is_alert
    if not observed_source_rows.all():
        raise AssertionError("At least one raw source row has no product detection")

    product_scores = product_scores_by_source[validation_to_source]
    product_alerts = product_alerts_by_source[validation_to_source]
    reference_alerts = reference_scores >= FLOW_THRESHOLD
    score_differences = np.abs(product_scores - reference_scores)
    threshold_disagreements = int(np.count_nonzero(product_alerts != reference_alerts))

    reference_flows = identity_source.iloc[validation_to_source].reset_index(drop=True)
    reference_flows["validation_row"] = np.arange(EXPECTED_ROWS, dtype=np.int64)
    reference_flows["attack_score"] = reference_scores
    reference_flows["is_alert"] = reference_alerts
    reference_incidents = reference_incident_table(reference_flows)
    product_incidents = product_incident_table(
        product.incidents, source_to_validation
    )
    incident_report = compare_incident_tables(product_incidents, reference_incidents)
    product_promoted = int(product_incidents["promoted"].sum())
    reference_promoted = int(reference_incidents["promoted"].sum())

    report = {
        "verification": "full frozen product detector operational parity",
        "capture_date": "2015-02-17",
        "flow_layer": {
            "flows_processed": product.flows_processed,
            "product_flow_alerts": product.flow_alert_count,
            "reference_flow_alerts": int(reference_alerts.sum()),
            "expected_flow_alerts": EXPECTED_FLOW_ALERTS,
            "maximum_absolute_score_difference": float(
                score_differences.max(initial=0.0)
            ),
            "score_differences_gt_1e_12": int(
                np.count_nonzero(score_differences > 1e-12)
            ),
            "threshold": FLOW_THRESHOLD,
            "threshold_decision_disagreements": threshold_disagreements,
        },
        "incident_layer": {
            **incident_report,
            "expected_incidents": EXPECTED_INCIDENTS,
            "policy": "B",
            "grouping_key": list(INCIDENT_KEYS),
            "window_seconds": INCIDENT_WINDOW_SECONDS,
            "boundary": "same incident when gap <= 300s; new incident when gap > 300s",
        },
        "promotion_layer": {
            "threshold": PROMOTION_THRESHOLD,
            "product_promoted_incidents": product_promoted,
            "reference_promoted_incidents": reference_promoted,
            "expected_promoted_incidents": EXPECTED_PROMOTED_INCIDENTS,
            "promotion_decision_disagreements": incident_report[
                "promotion_decision_disagreements"
            ],
        },
        "source_alignment": alignment_report,
        "product_contract": {
            "product_input_column_count": len(product_input_columns),
            "ground_truth_columns_removed_before_detector": (
                removed_evaluation_columns
            ),
            "labels_entered_product_path": False,
            "model_version": product.model_version,
            "feature_contract": product.feature_contract,
            "state_mode": product.state_mode,
        },
        "frozen_decisions": {
            "model_retrained": False,
            "model_retuned": False,
            "threshold_changed": False,
            "aggregation_policy_changed": False,
            "promotion_policy_changed": False,
            "holdout_accessed": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    required_zeroes = (
        report["flow_layer"]["score_differences_gt_1e_12"],
        report["flow_layer"]["threshold_decision_disagreements"],
        report["incident_layer"]["missing_reference_memberships"],
        report["incident_layer"]["extra_product_memberships"],
        report["incident_layer"]["grouping_key_disagreements"],
        report["incident_layer"]["first_last_timestamp_disagreements"],
        report["incident_layer"]["flow_count_disagreements"],
        report["incident_layer"]["max_score_disagreements_gt_1e_12"],
        report["incident_layer"]["mean_score_disagreements_gt_1e_12"],
        report["promotion_layer"]["promotion_decision_disagreements"],
    )
    if any(required_zeroes):
        raise AssertionError(f"Product operational parity failed: {report}")
    if (
        product.flow_alert_count,
        product.incident_count,
        product.promoted_incident_count,
    ) != (EXPECTED_FLOW_ALERTS, EXPECTED_INCIDENTS, EXPECTED_PROMOTED_INCIDENTS):
        raise AssertionError("Frozen Feb 17 operational counts changed")
    if (
        int(reference_alerts.sum()),
        len(reference_incidents),
        reference_promoted,
    ) != (EXPECTED_FLOW_ALERTS, EXPECTED_INCIDENTS, EXPECTED_PROMOTED_INCIDENTS):
        raise AssertionError("Frozen research reference counts changed")
    if args.report_path:
        write_report(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
