from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src.security_anomaly import cli
from src.security_anomaly.detector import SecurityAnomalyDetector
from src.security_anomaly.serialization import jsonl_text, promoted_incidents_to_v1
from src.security_anomaly.validation import read_and_validate_flow_csv
from tools.generate_product_fixture import write_fixture


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "product-v01"
FLOW_FIXTURE = FIXTURE_DIR / "flows.csv"
EXPECTED_JSONL = FIXTURE_DIR / "expected-incidents.jsonl"
EXPECTED_RUN = FIXTURE_DIR / "expected-run.json"
MANIFEST = ROOT / "contracts" / "model-manifest-context-rf-v2.json"
CONTRACT = ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
SCHEMA = ROOT / "contracts" / "incident-v1.schema.json"
MODEL_ENV = "SECURITY_ANOMALY_E2E_MODEL"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_run() -> dict[str, object]:
    return json.loads(EXPECTED_RUN.read_text(encoding="utf-8"))


def test_synthetic_fixture_is_reproducible_label_free_and_redistributable(
    tmp_path: Path,
) -> None:
    regenerated = tmp_path / "flows.csv"
    write_fixture(regenerated)
    assert regenerated.read_bytes() == FLOW_FIXTURE.read_bytes()

    expected = expected_run()
    assert sha256(FLOW_FIXTURE) == expected["fixture"]["sha256"]
    assert expected["fixture"]["fully_synthetic"] is True
    assert expected["fixture"]["contains_research_dataset_rows"] is False
    batch = read_and_validate_flow_csv(FLOW_FIXTURE)
    assert batch.row_count == expected["fixture"]["rows"] == 12
    assert not {"Label", "label", "attack_cat"} & set(batch.frame.columns)
    assert batch.frame["Timestamp"].duplicated().any()
    assert batch.frame["Src IP"].nunique() > 1
    assert batch.frame["Dst IP"].nunique() > 1
    assert sorted(batch.frame["Protocol"].unique().tolist()) == [6, 17]


def test_committed_golden_is_strict_incident_v1_and_freezes_versions() -> None:
    expected = expected_run()
    assert sha256(EXPECTED_JSONL) == expected["expected_output"]["sha256"]
    assert expected["versions"] == {
        "product_version": "0.1.0",
        "model_version": "context-rf-v2",
        "model_sha256": (
            "4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72"
        ),
        "feature_contract": "cicflow-v2-128",
        "feature_builder": "causal-temporal-v2",
        "incident_contract": "incident-v1",
        "model_release_tag": "model-context-rf-v2",
    }
    assert expected["frozen_policy"] == {
        "flow_threshold": 0.1,
        "flow_operator": ">=",
        "aggregation_policy": "B",
        "grouping_key": ["src_ip", "dst_ip", "dst_port"],
        "incident_window_seconds": 300,
        "new_incident_gap_operator": ">",
        "promotion_threshold": 0.25,
        "promotion_operator": ">=",
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    objects = [json.loads(line) for line in EXPECTED_JSONL.read_text().splitlines()]
    assert len(objects) == expected["expected_counts"]["promoted_incidents"] == 2
    for value in objects:
        jsonschema.validate(value, schema)
        assert value["promoted"] is True
        assert not {"Label", "label", "attack_cat"} & set(value)


def _real_model_or_skip() -> Path:
    configured = os.environ.get(MODEL_ENV)
    if not configured:
        pytest.skip(f"Set {MODEL_ENV} to run the public frozen-model E2E gate")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"{MODEL_ENV} does not point to a model file: {path}")
    return path


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_real_frozen_model_full_product_path_matches_golden(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = _real_model_or_skip()
    expected = expected_run()
    assert sha256(model_path) == expected["versions"]["model_sha256"]

    detector = SecurityAnomalyDetector.load(model_path, MANIFEST, CONTRACT)
    batch = read_and_validate_flow_csv(FLOW_FIXTURE, contract=detector.bundle.contract)
    first = detector.analyze_batch(batch.frame)
    second = detector.analyze_batch(batch.frame)
    assert (
        first.flows_processed,
        first.flow_alert_count,
        first.incident_count,
        first.promoted_incident_count,
    ) == (
        second.flows_processed,
        second.flow_alert_count,
        second.incident_count,
        second.promoted_incident_count,
    )
    assert [
        (flow.source_row_id, flow.flow_id, flow.is_alert)
        for flow in first.flow_detections
    ] == [
        (flow.source_row_id, flow.flow_id, flow.is_alert)
        for flow in second.flow_detections
    ]

    counts = expected["expected_counts"]
    assert first.flows_processed == counts["flows_processed"]
    assert first.flow_alert_count == counts["flow_alerts"]
    assert first.incident_count == counts["incidents"]
    assert first.promoted_incident_count == counts["promoted_incidents"]

    expected_scores = {
        int(row_id): score
        for row_id, score in expected["score_regression"][
            "scores_by_source_row_id"
        ].items()
    }
    actual_scores = {
        flow.source_row_id: flow.attack_score for flow in first.flow_detections
    }
    assert actual_scores.keys() == expected_scores.keys()
    np.testing.assert_allclose(
        [actual_scores[index] for index in sorted(actual_scores)],
        [expected_scores[index] for index in sorted(expected_scores)],
        rtol=0,
        atol=expected["score_regression"]["absolute_tolerance"],
    )

    actual_incidents = [
        {
            "src_ip": item.src_ip,
            "dst_ip": item.dst_ip,
            "dst_port": item.dst_port,
            "first_seen": item.first_seen.isoformat(timespec="seconds"),
            "last_seen": item.last_seen.isoformat(timespec="seconds"),
            "protocols": list(item.protocols),
            "source_row_ids": list(item.source_row_ids),
            "flow_count": item.flow_count,
            "promoted": item.promoted,
        }
        for item in first.incidents
    ]
    assert actual_incidents == expected["incident_regression"]

    direct_jsonl = jsonl_text(promoted_incidents_to_v1(first)).encode("utf-8")
    repeated_jsonl = jsonl_text(promoted_incidents_to_v1(second)).encode("utf-8")
    assert direct_jsonl == repeated_jsonl
    assert direct_jsonl == EXPECTED_JSONL.read_bytes()

    cli_outputs = []
    for run in (1, 2):
        output = tmp_path / f"incidents-{run}.jsonl"
        assert cli.main(
            [
                "analyze",
                str(FLOW_FIXTURE),
                "--model",
                str(model_path),
                "--output",
                str(output),
            ]
        ) == 0
        summary = json.loads(capsys.readouterr().err)
        assert summary["flows_processed"] == first.flows_processed
        assert summary["flow_alerts"] == first.flow_alert_count
        assert summary["aggregated_incidents"] == first.incident_count
        assert summary["promoted_incidents"] == first.promoted_incident_count
        cli_outputs.append(output.read_bytes())
    assert cli_outputs[0] == cli_outputs[1] == direct_jsonl
