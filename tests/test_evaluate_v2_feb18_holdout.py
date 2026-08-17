from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd

from src import evaluate_v2_feb18_holdout as final_holdout


def test_frozen_operational_constants_are_exact() -> None:
    assert final_holdout.FLOW_THRESHOLD == 0.10
    assert final_holdout.INCIDENT_KEYS == ("src_ip", "dst_ip", "dst_port")
    assert final_holdout.INCIDENT_WINDOW_SECONDS == 300
    assert final_holdout.PROMOTION_SCORE_THRESHOLD == 0.25
    assert final_holdout.EXPECTED_HOLDOUT_ROWS == 1_275_429


def test_final_evaluator_does_not_fit_or_retrain_a_model() -> None:
    source = inspect.getsource(final_holdout)
    assert "model.fit(" not in source
    assert "RandomForestClassifier(" not in source
    assert "GridSearchCV(" not in source
    assert "RandomizedSearchCV(" not in source


def test_binary_flow_metrics_use_locked_threshold_and_normalized_rates() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.01, 0.10, 0.09, 0.80])
    metrics = final_holdout.binary_flow_metrics(labels, scores, capture_hours=2.0)

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["alerts_per_hour"] == 1.0
    assert metrics["fp_flows_per_hour"] == 0.5


def test_drift_helpers_report_shift_and_unseen_categories() -> None:
    reference = np.arange(1, 101, dtype=float)
    shifted = reference + 100.0
    assert final_holdout.population_stability_index(reference, reference) == 0.0
    assert final_holdout.population_stability_index(reference, shifted) > 0.0

    drift = final_holdout.categorical_drift(["tcp", "tcp", "udp"], ["tcp", "icmp"])
    assert drift["unseen_level_count"] == 1
    assert drift["unseen_levels_sample"] == ["icmp"]
    assert drift["jensen_shannon_divergence_bits"] > 0.0


def test_temporal_change_labels_respect_metric_direction() -> None:
    labels = final_holdout.classify_temporal_changes(
        {
            "flow_recall": 0.01,
            "flow_fpr": -0.01,
            "promotion_incident_precision": -0.01,
            "promotion_fp_incidents_per_hour": 0.01,
            "flow_pr_auc": 0.0005,
        }
    )
    assert labels["flow_recall"]["status"] == "improved"
    assert labels["flow_fpr"]["status"] == "improved"
    assert labels["promotion_incident_precision"]["status"] == "degraded"
    assert labels["promotion_fp_incidents_per_hour"]["status"] == "degraded"
    assert labels["flow_pr_auc"]["status"] == "stable"


def test_holdout_router_uses_timestamp_only_and_preserves_selected_bytes(
    tmp_path, monkeypatch
) -> None:
    columns = [f"column_{index}" for index in range(84)]
    columns[6] = "Timestamp"

    def row(timestamp: str, marker: str) -> bytes:
        values = [marker] * 84
        values[6] = timestamp
        return (",".join(values) + "\n").encode()

    header = (",".join(columns) + "\n").encode()
    keep_a = row("18/02/2015 12:00:00 AM", "keep-a")
    skip = row("17/02/2015 11:59:59 PM", "skip")
    keep_b = row("18/02/2015 12:00:01 AM", "keep-b")
    raw_path = tmp_path / "raw.csv"
    raw_path.write_bytes(header + keep_a + skip + keep_b)
    cache_dir = tmp_path / "cache"
    routed_path = cache_dir / "feb18.csv"
    monkeypatch.setattr(final_holdout, "RAW_PATH", raw_path)
    monkeypatch.setattr(final_holdout, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(final_holdout, "ROUTED_HOLDOUT_CSV", routed_path)
    monkeypatch.setattr(final_holdout, "EXPECTED_HOLDOUT_ROWS", 2)

    result = final_holdout.route_holdout_once()

    expected = header + keep_a + keep_b
    assert routed_path.read_bytes() == expected
    assert result["rows"] == 2
    assert result["materialization_count"] == 1
    assert result["materialized_non_holdout_rows"] == 0
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()
