from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src import analyze_incident_promotion
from src.analyze_incident_promotion import (
    build_incident_evidence,
    candidate_grid,
    evaluate_all_candidates,
    promoted_mask,
)


def fixture_flows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_row": range(8),
            "timestamp": pd.to_datetime(
                [
                    "2015-01-22 10:00:00",
                    "2015-01-22 10:00:10",
                    "2015-01-22 10:00:20",
                    "2015-01-22 10:00:30",
                    "2015-01-22 10:10:00",
                    "2015-01-22 10:10:01",
                    "2015-01-22 10:20:00",
                    "2015-01-22 10:20:01",
                ]
            ),
            "src_ip": ["a"] * 4 + ["c"] * 2 + ["e"] * 2,
            "dst_ip": ["b"] * 4 + ["d"] * 2 + ["f"] * 2,
            "dst_port": [80] * 4 + [53] * 2 + [443] * 2,
            "attack_cat": ["Benign"] * 4 + ["Fuzzers"] * 4,
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "attack_score": [0.11, 0.20, 0.30, 0.50, 0.40, 0.05, 0.60, 0.05],
            "predicted_class": [1, 1, 1, 1, 1, 0, 1, 0],
        }
    )


def test_incident_evidence_uses_scores_and_counts_without_ground_truth() -> None:
    evidence, membership, truth = build_incident_evidence(fixture_flows())
    benign_incident = evidence.loc[evidence["alert_flow_count"] == 4].iloc[0]

    assert "label" not in evidence
    assert "attack_cat" not in evidence
    assert benign_incident["duration_seconds"] == 30
    assert benign_incident["max_attack_score"] == 0.50
    assert benign_incident["flows_score_ge_0_30"] == 2
    assert np.isclose(benign_incident["top3_mean_attack_score"], 1.0 / 3.0)
    assert len(membership) == 6
    assert int(truth["pure_false_positive"].sum()) == 1


def test_candidate_grid_is_exactly_the_predeclared_42_rules() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 42
    assert sum(candidate["family"] == "A" for candidate in candidates) == 17
    assert sum(candidate["family"] == "B" for candidate in candidates) == 16
    assert sum(candidate["family"] == "C" for candidate in candidates) == 9
    assert {candidate["score_threshold"] for candidate in candidates if candidate["family"] == "A"} == set(
        np.round(np.arange(0.10, 0.9001, 0.05), 2)
    )


def test_family_b_and_c_masks_follow_the_declared_or_rules() -> None:
    evidence = pd.DataFrame(
        {
            "max_attack_score": [0.1, 0.4, 0.1],
            "alert_flow_count": [5, 1, 1],
            "flows_score_ge_0_30": [0, 1, 3],
        }
    )
    family_b = {
        "family": "B",
        "score_threshold": 0.3,
        "count_threshold": 5,
        "count_field": "alert_flow_count",
    }
    family_c = {
        "family": "C",
        "score_threshold": 0.5,
        "count_threshold": 3,
        "count_field": "flows_score_ge_0_30",
    }
    assert promoted_mask(evidence, family_b).tolist() == [True, True, False]
    assert promoted_mask(evidence, family_c).tolist() == [False, False, True]


def test_selection_prefers_minimum_fp_after_recall_gate() -> None:
    evidence, membership, truth = build_incident_evidence(fixture_flows())
    # Two attack reference incidents; both are detected by their one alert row.
    reference = pd.DataFrame(
        {
            "validation_row": [4, 5, 6, 7],
            "reference_incident_id": ["r1", "r1", "r2", "r2"],
        }
    )
    results, selected, gate = evaluate_all_candidates(
        evidence, membership, truth, reference, capture_hours=1.0
    )
    assert gate
    assert selected["incident_recall"] == 1.0
    eligible = results.loc[results["recall_requirement_achieved"]]
    assert selected["promoted_pure_fp_incidents"] == eligible[
        "promoted_pure_fp_incidents"
    ].min()


def test_promotion_module_has_no_holdout_path_and_no_model_training() -> None:
    source = inspect.getsource(analyze_incident_promotion)
    forbidden_name = "locked" + "_holdout_features.parquet"
    assert forbidden_name not in source
    assert ".fit(" not in source

