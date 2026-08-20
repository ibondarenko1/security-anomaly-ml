from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.security_anomaly import cli
from src.security_anomaly.results import BatchAnalysisResult, IncidentDetection
from src.security_anomaly.serialization import jsonl_text, promoted_incidents_to_v1


def fake_result(*, empty: bool = False) -> BatchAnalysisResult:
    incidents = ()
    if not empty:
        incidents = (
            IncidentDetection(
                incident_sequence=0,
                first_seen=datetime(2015, 2, 17, 20, 0, 0),
                last_seen=datetime(2015, 2, 17, 20, 0, 1),
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                dst_port=443,
                protocols=(6, 17),
                flow_count=2,
                max_attack_score=0.4,
                mean_attack_score=0.3,
                promoted=True,
                source_row_ids=(0, 1),
            ),
        )
    return BatchAnalysisResult(
        flows_processed=0 if empty else 2,
        flow_alert_count=0 if empty else 2,
        incident_count=len(incidents),
        promoted_incident_count=len(incidents),
        flow_detections=(),
        incidents=incidents,
        promoted_incidents=incidents,
        product_version="0.1.0",
        model_version="context-rf-v2",
        feature_contract="cicflow-v2-128",
        state_mode="batch-empty",
    )


class FakeDetector:
    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def analyze_batch(self, frame: pd.DataFrame) -> BatchAnalysisResult:
        self.seen = frame.copy()
        return fake_result(empty=frame.empty)


def write_fixture(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def test_help_and_version_work_without_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        cli.main(["--help"])
    assert help_exit.value.code == 0
    assert "security-anomaly analyze" in capsys.readouterr().out

    assert cli.main(["version"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "product_version": "0.1.0",
        "feature_contract": "cicflow-v2-128",
        "feature_builder_version": "causal-temporal-v2",
        "incident_contract": "incident-v1",
    }


def test_validate_is_label_free_and_does_not_load_model(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "flows.csv"
    write_fixture(source, valid_flow_frame)
    monkeypatch.setattr(cli, "_load_detector", lambda _path: pytest.fail("loaded model"))
    assert cli.main(["validate", str(source)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 2
    assert payload["timestamp_min"] == "2015-02-17T20:00:00"


def test_analyze_uses_shared_serializer_and_removes_labels(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "flows.csv"
    write_fixture(source, valid_flow_frame.assign(Label=["Benign", "Attack"]))
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test")
    output = tmp_path / "incidents.jsonl"
    detector = FakeDetector()
    monkeypatch.setattr(cli, "_load_detector", lambda _path: detector)

    assert cli.main(
        ["analyze", str(source), "--model", str(model), "--output", str(output)]
    ) == 0
    assert detector.seen is not None
    assert "Label" not in detector.seen
    expected = jsonl_text(promoted_incidents_to_v1(fake_result()))
    assert output.read_text(encoding="utf-8") == expected
    summary = json.loads(capsys.readouterr().err)
    assert summary["validation_status"] == "valid"
    assert summary["rejected_rows"] == 0
    assert summary["promoted_incidents"] == 1
    assert summary["incident_contract"] == "incident-v1"


def test_header_only_analyze_writes_successful_empty_jsonl(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "empty.csv"
    write_fixture(source, valid_flow_frame.head(0))
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test")
    output = tmp_path / "incidents.jsonl"
    detector = FakeDetector()
    monkeypatch.setattr(cli, "_load_detector", lambda _path: detector)
    assert cli.main(
        ["analyze", str(source), "--model", str(model), "--output", str(output)]
    ) == 0
    assert output.read_bytes() == b""


def test_missing_and_corrupt_model_have_stable_exit_code(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "flows.csv"
    write_fixture(source, valid_flow_frame)
    output = tmp_path / "out.jsonl"
    missing = tmp_path / "missing.joblib"
    assert cli.main(
        ["analyze", str(source), "--model", str(missing), "--output", str(output)]
    ) == cli.EXIT_MODEL
    assert "model error" in capsys.readouterr().err

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not-the-frozen-model")
    assert cli.main(
        ["analyze", str(source), "--model", str(corrupt), "--output", str(output)]
    ) == cli.EXIT_MODEL
    assert "SHA256 mismatch" in capsys.readouterr().err


def test_malformed_input_and_output_failure_have_stable_exit_codes(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("A,A\n1,2\n", encoding="utf-8")
    assert cli.main(["validate", str(malformed)]) == cli.EXIT_INPUT
    assert "duplicate columns" in capsys.readouterr().err

    source = tmp_path / "flows.csv"
    write_fixture(source, valid_flow_frame)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test")
    monkeypatch.setattr(cli, "_load_detector", lambda _path: FakeDetector())
    impossible = tmp_path / "missing-parent" / "out.jsonl"
    assert cli.main(
        ["analyze", str(source), "--model", str(model), "--output", str(impossible)]
    ) == cli.EXIT_OUTPUT
    assert "output error" in capsys.readouterr().err
    assert not impossible.exists()


def test_model_info_reports_frozen_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test")
    metadata = {
        "product_version": "0.1.0",
        "model_version": "context-rf-v2",
        "model_sha256": "abc",
        "feature_contract": "cicflow-v2-128",
        "feature_builder_version": "causal-temporal-v2",
        "serialization": {"python_major_minor": "3.13"},
    }
    fake = SimpleNamespace(
        bundle=SimpleNamespace(metadata=lambda: metadata),
        policy=SimpleNamespace(
            flow_threshold=0.10,
            incident_policy="B",
            grouping_key=("src_ip", "dst_ip", "dst_port"),
            incident_window_seconds=300,
            promotion_threshold=0.25,
        ),
    )
    monkeypatch.setattr(cli, "_load_detector", lambda _path: fake)
    assert cli.main(["model-info", "--model", str(model)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["flow_threshold"] == 0.10
    assert payload["aggregation_policy"] == "B"
    assert payload["promotion_threshold"] == 0.25


def test_model_environment_variable_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_model = tmp_path / "environment.joblib"
    environment_model.write_bytes(b"test")
    monkeypatch.setenv(cli.MODEL_ENVIRONMENT_VARIABLE, str(environment_model))
    assert cli.resolve_model_path(None) == environment_model.resolve()
