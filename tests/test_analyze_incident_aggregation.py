from __future__ import annotations

import inspect

import pandas as pd

from src import analyze_incident_aggregation
from src.analyze_incident_aggregation import (
    assign_incidents,
    incident_table,
    reference_attack_metrics,
    select_operational_candidates,
)


def fixture_flows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_row": range(7),
            "timestamp": pd.to_datetime(
                [
                    "2015-02-17 10:00:00",
                    "2015-02-17 10:00:10",
                    "2015-02-17 10:00:20",
                    "2015-02-17 10:00:40",
                    "2015-02-17 10:00:40",
                    "2015-02-17 10:02:00",
                    "2015-02-17 10:02:00",
                ]
            ),
            "src_ip": ["a", "x", "a", "a", "a", "a", "a"],
            "dst_ip": ["b", "y", "b", "b", "b", "b", "b"],
            "dst_port": [80, 53, 80, 80, 80, 80, 80],
            "protocol": [6, 17, 6, 6, 6, 6, 6],
            "attack_cat": [
                "Benign",
                "Benign",
                "Fuzzers",
                "Benign",
                "Fuzzers",
                "Fuzzers",
                "Fuzzers",
            ],
            "label": [0, 0, 1, 0, 1, 1, 1],
            "attack_score": [0.2, 0.2, 0.8, 0.3, 0.9, 0.8, 0.01],
            "predicted_class": [1, 1, 1, 1, 1, 1, 0],
        }
    )


def test_incidents_use_consecutive_gap_per_key_and_keep_same_second_together() -> None:
    assigned = assign_incidents(fixture_flows(), ["src_ip", "dst_ip"], 30)
    target = assigned.loc[(assigned["src_ip"] == "a") & (assigned["dst_ip"] == "b")]

    # The unrelated x/y row at t+10 does not split a/b.  Gaps of exactly 20 and
    # 30 seconds stay together, same-second peers stay together, and 80 seconds splits.
    assert target.iloc[0]["incident_id"] == target.iloc[1]["incident_id"]
    assert target.iloc[1]["incident_id"] == target.iloc[2]["incident_id"]
    assert target.iloc[2]["incident_id"] == target.iloc[3]["incident_id"]
    assert target.iloc[3]["incident_id"] != target.iloc[4]["incident_id"]
    assert target.iloc[4]["incident_id"] == target.iloc[5]["incident_id"]


def test_incident_labels_are_exclusive_but_true_positive_includes_mixed() -> None:
    alerts = fixture_flows().loc[lambda frame: frame["predicted_class"] == 1]
    incidents = incident_table(alerts, ["src_ip", "dst_ip"], 30)

    assert len(incidents) == 3
    assert int(incidents["false_positive"].sum()) == 1
    assert int(incidents["mixed"].sum()) == 1
    assert int(incidents["attack_only"].sum()) == 1
    assert int(incidents["true_positive_including_mixed"].sum()) == 2


def test_reference_incident_is_detected_when_any_member_flow_alerts() -> None:
    metrics = reference_attack_metrics(
        fixture_flows(), ["src_ip", "dst_ip"], 30, "A"
    )
    fuzzers = metrics.loc[metrics["attack_category"] == "Fuzzers"].iloc[0]

    assert fuzzers["attack_flows"] == 4
    assert fuzzers["detected_attack_flows"] == 3
    assert fuzzers["attack_incidents"] == 2
    assert fuzzers["detected_attack_incidents"] == 2
    assert fuzzers["incident_recall"] == 1.0


def test_analysis_module_has_no_holdout_dataset_path_or_model_fit() -> None:
    source = inspect.getsource(analyze_incident_aggregation)
    forbidden_name = "locked" + "_holdout_features.parquet"
    assert forbidden_name not in source
    assert ".fit(" not in source


def test_candidate_selection_rejects_long_coarse_key_and_prefers_service_aware_b() -> None:
    rows = []
    for policy, key_count in (("A", 2), ("B", 3), ("C", 4)):
        for window, fp in ((60, 300), (300, 80 if policy == "A" else 900)):
            rows.append(
                {
                    "policy": policy,
                    "policy_definition": policy,
                    "grouping_key_count": key_count,
                    "window_seconds": window,
                    "window_label": str(window),
                    "false_positive_incidents": fp + (20 if policy == "C" else 0),
                    "fp_compression_ratio": 10.0,
                    "alert_reduction_percentage": 0.8,
                    "overall_attack_incident_recall": 0.999,
                    "normal_alert_absorption_into_mixed_fraction": 0.5,
                    "operational_candidate": False,
                    "candidate_rank": float("nan"),
                    "recommended": False,
                }
            )
    selected, candidates = select_operational_candidates(pd.DataFrame(rows))

    selected_a = selected.loc[
        (selected["policy"] == "A") & selected["operational_candidate"]
    ].iloc[0]
    recommended = next(row for row in candidates if row["recommended"])
    assert selected_a["window_seconds"] == 60
    assert recommended["policy"] == "B"
