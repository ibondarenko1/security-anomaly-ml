from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.security_anomaly.replay as replay
from src.security_anomaly.contracts import FeatureContract
from src.security_anomaly.detector import FrozenOperationalPolicy, SecurityAnomalyDetector
from src.security_anomaly.serialization import promoted_incidents_to_v1
from src.security_anomaly.temporal import CausalTemporalFeatureBuilder
from src.security_anomaly.validation import InputContractError, validate_flow_records


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contracts" / "model-manifest-context-rf-v2.json"
CONTRACT = ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
FIXTURE = ROOT / "tests" / "fixtures" / "product-v01" / "flows.csv"


class _FakeBundle:
    """Deterministic scores while retaining the real temporal feature builder."""

    def __init__(self, contract: FeatureContract) -> None:
        self.contract = contract
        self.model_version = "context-rf-v2"
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self._lifetime_column = contract.model_features.index("src_conn_60s")

    def metadata(self) -> dict[str, object]:
        return {
            "product_version": self.manifest["product_version"],
            "model_version": self.model_version,
            "feature_contract": self.contract.version,
            "feature_builder_version": self.contract.builder_version,
            "model_sha256": "bound-by-replay-file-digest",
        }

    def predict_scores(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.ndim == 2 and matrix.shape[1] == 128
        lifetime = matrix[:, self._lifetime_column]
        return np.clip(0.30 + lifetime * 0.01, 0.0, 1.0)


def _fake_detector() -> SecurityAnomalyDetector:
    contract = FeatureContract.load()
    bundle = _FakeBundle(contract)
    return SecurityAnomalyDetector(
        bundle,  # type: ignore[arg-type]
        CausalTemporalFeatureBuilder(contract),
        FrozenOperationalPolicy.from_bundle(bundle),  # type: ignore[arg-type]
    )


def _configure_fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    model = tmp_path / "fake-model.joblib"
    model.write_bytes(b"replay-test-model")
    detector = _fake_detector()
    monkeypatch.setattr(replay, "_resolve_model_path", lambda _explicit: model)
    monkeypatch.setattr(replay, "_load_detector", lambda _path: detector)
    return model


def _flow_frame(valid_flow_frame: pd.DataFrame, count: int) -> pd.DataFrame:
    frame = pd.concat([valid_flow_frame.head(1)] * count, ignore_index=True)
    frame["Flow ID"] = [f"flow-{number}" for number in range(count)]
    frame["Src IP"] = ["192.0.2.10"] * count
    frame["Dst IP"] = ["198.51.100.20"] * count
    frame["Src Port"] = [50000 + number for number in range(count)]
    frame["Dst Port"] = [445] * count
    frame["Protocol"] = [6 if number % 2 == 0 else 17 for number in range(count)]
    frame["Timestamp"] = [
        "20/08/2026 01:00:00 PM",
        "20/08/2026 01:00:00 PM",
        "20/08/2026 01:00:05 PM",
        "20/08/2026 01:01:00 PM",
        "20/08/2026 01:01:00 PM",
        "20/08/2026 01:06:00 PM",
    ][:count]
    return frame


def _write_rotation(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def _run(
    directory: Path,
    checkpoint: Path,
    output: Path,
    *,
    model: Path,
    timezone: str = "UTC",
    sensor: str = "sensor-a",
):
    return replay.replay_directory(
        directory,
        model=model,
        sensor=sensor,
        timezone=timezone,
        checkpoint=checkpoint,
        output=output,
    )


def test_full_and_late_split_replay_are_byte_equivalent_across_equal_times(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"
    full_dir.mkdir()
    split_dir.mkdir()
    full = _flow_frame(valid_flow_frame, 6)
    _write_rotation(full_dir, "all.csv", full)
    _write_rotation(split_dir, "02-late.csv", full.iloc[3:].reset_index(drop=True))

    full_output = tmp_path / "full.jsonl"
    _run(full_dir, tmp_path / "full.sqlite", full_output, model=model)

    split_output = tmp_path / "split.jsonl"
    split_checkpoint = tmp_path / "split.sqlite"
    first = _run(split_dir, split_checkpoint, split_output, model=model)
    assert first.rescored is True
    before_late = split_output.read_bytes()

    _write_rotation(split_dir, "01-early.csv", full.iloc[:3].reset_index(drop=True))
    second = _run(split_dir, split_checkpoint, split_output, model=model)
    assert second.rotations_added == 1
    assert second.rescored is True
    assert split_output.read_bytes() != before_late
    assert split_output.read_bytes() == full_output.read_bytes()


def test_unchanged_restart_skips_model_and_missing_output_recovers_snapshot(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    checkpoint = tmp_path / "replay.sqlite"
    output = tmp_path / "incidents.jsonl"
    _run(directory, checkpoint, output, model=model)
    expected = output.read_bytes()

    monkeypatch.setattr(
        replay,
        "_load_detector",
        lambda _path: pytest.fail("unchanged replay loaded the model"),
    )
    unchanged = _run(directory, checkpoint, output, model=model)
    assert unchanged.status == "unchanged"
    assert unchanged.rescored is False

    output.unlink()
    recovered = _run(directory, checkpoint, output, model=model)
    assert recovered.status == "recovered"
    assert recovered.rescored is False
    assert output.read_bytes() == expected


def test_modified_or_invalid_rotation_rolls_back_without_losing_accepted_data(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    accepted = _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    checkpoint = tmp_path / "replay.sqlite"
    output = tmp_path / "incidents.jsonl"
    _run(directory, checkpoint, output, model=model)
    output_before = output.read_bytes()

    changed = _flow_frame(valid_flow_frame, 2)
    changed.loc[0, "Flow ID"] = "modified-after-acceptance"
    changed.to_csv(accepted, index=False)
    with pytest.raises(InputContractError, match="modified"):
        _run(directory, checkpoint, output, model=model)
    assert output.read_bytes() == output_before

    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    invalid = _flow_frame(valid_flow_frame, 1)
    invalid.loc[0, "Src IP"] = "not-an-ip"
    _write_rotation(directory, "02-invalid.csv", invalid)
    with pytest.raises(InputContractError, match="valid IPv4/IPv6"):
        _run(directory, checkpoint, output, model=model)
    assert output.read_bytes() == output_before
    with sqlite3.connect(checkpoint) as connection:
        rows = connection.execute(
            "SELECT rotation_id FROM replay_rotations ORDER BY rotation_id"
        ).fetchall()
    assert rows == [("01.csv",)]


def test_renamed_bytes_deduplicate_and_deleted_source_rollover_is_retained(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    original = _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    checkpoint = tmp_path / "replay.sqlite"
    output = tmp_path / "incidents.jsonl"
    first = _run(directory, checkpoint, output, model=model)
    expected = output.read_bytes()
    original.rename(directory / "02-renamed.csv")

    renamed = _run(directory, checkpoint, output, model=model)
    assert renamed.status == "unchanged"
    assert renamed.rescored is False
    assert renamed.rotations == 1
    assert output.read_bytes() == expected

    (directory / "02-renamed.csv").unlink()
    rollover = _run(directory, checkpoint, output, model=model)
    assert rollover.status == "unchanged"
    assert rollover.rescored is False
    assert rollover.rotations == 1
    assert output.read_bytes() == expected


def test_unrelated_nonempty_sqlite_checkpoint_is_rejected(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 1))
    checkpoint = tmp_path / "foreign.sqlite"
    with sqlite3.connect(checkpoint) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    with pytest.raises(InputContractError, match="supported replay checkpoint|supported"):
        _run(directory, checkpoint, tmp_path / "incidents.jsonl", model=model)
    with sqlite3.connect(checkpoint) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchone() == ("keep",)


def test_cli_ocsf_format_exports_sensor_namespace(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    import src.security_anomaly.cli as cli

    output = tmp_path / "findings.jsonl"
    assert (
        cli.main(
            [
                "replay",
                str(directory),
                "--model",
                str(model),
                "--sensor",
                "sensor-cli",
                "--timezone",
                "UTC",
                "--format",
                "ocsf",
                "--checkpoint",
                str(tmp_path / "ocsf.sqlite"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert records[0]["metadata"]["log_provider"] == "sensor-cli"


def test_interrupted_publication_recovers_committed_snapshot_after_source_rollover(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 1))
    checkpoint = tmp_path / "replay.sqlite"
    output = tmp_path / "incidents.jsonl"
    _run(directory, checkpoint, output, model=model)
    old_output = output.read_bytes()

    _write_rotation(directory, "02.csv", _flow_frame(valid_flow_frame, 2))
    original_atomic = replay.atomic_text_output

    def fail_publication(_path):
        raise replay.OutputWriteError("simulated interrupted publication")

    monkeypatch.setattr(replay, "atomic_text_output", fail_publication)
    with pytest.raises(replay.OutputWriteError, match="interrupted"):
        _run(directory, checkpoint, output, model=model)
    assert output.read_bytes() == old_output

    # Both accepted rotations are now in the committed checkpoint.  Removing the
    # source directory contents must not lose the committed newer snapshot.
    for path in directory.glob("*.csv"):
        path.unlink()
    monkeypatch.setattr(replay, "atomic_text_output", original_atomic)
    recovered = _run(directory, checkpoint, output, model=model)
    assert recovered.status == "recovered"
    assert recovered.rescored is False
    assert output.read_bytes() != old_output


def test_concurrent_writers_serialize_on_one_checkpoint(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    monkeypatch.setattr(replay, "_load_detector", lambda _path: _fake_detector())
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 2))
    checkpoint = tmp_path / "concurrent.sqlite"
    output = tmp_path / "concurrent.jsonl"

    def invoke():
        return _run(directory, checkpoint, output, model=model)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: invoke(), (1, 2)))
    assert sorted(result.status for result in results) == ["replayed", "unchanged"]
    assert output.is_file()
    assert output.read_bytes().endswith(b"\n")


def test_header_only_and_mixed_label_rotations_remain_label_free(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    mixed = _flow_frame(valid_flow_frame, 2).assign(
        Label=["Benign", "Attack"], attack_cat=["", "X"]
    )
    _write_rotation(directory, "01-mixed.csv", mixed)
    _write_rotation(directory, "02-header.csv", mixed.head(0))
    output = tmp_path / "incidents.jsonl"
    result = _run(
        directory,
        tmp_path / "label-free.sqlite",
        output,
        model=model,
    )
    assert result.rows == 2
    for line in output.read_text(encoding="utf-8").splitlines():
        assert "Label" not in line and "attack_cat" not in line


def test_alias_and_collision_paths_fail_closed(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    input_path = _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 1))
    same_path = tmp_path / "same.sqlite"
    with pytest.raises(Exception, match="differ"):
        _run(directory, same_path, same_path, model=model)

    output_alias = directory / "02.csv"
    output_alias.write_bytes(input_path.read_bytes())
    with pytest.raises(Exception, match="input rotation|different"):
        _run(directory, tmp_path / "alias.sqlite", output_alias, model=model)


def test_model_and_reserved_sqlite_sidecar_collisions_fail_closed(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    _write_rotation(directory, "01.csv", _flow_frame(valid_flow_frame, 1))
    checkpoint = tmp_path / "replay.sqlite"
    with pytest.raises(Exception, match="differ|rotation"):
        _run(directory, checkpoint, model, model=model)
    with pytest.raises(Exception, match="differ|rotation|sidecar"):
        _run(directory, Path(f"{checkpoint}-wal"), tmp_path / "output.jsonl", model=model)


def test_validation_canonicalizes_real_ipv4_ipv6_and_rejects_invalid_values(
    valid_flow_frame: pd.DataFrame,
) -> None:
    frame = valid_flow_frame.copy()
    frame.loc[0, "Src IP"] = "2001:0DB8:0000:0000:0000:0000:0000:0001"
    frame.loc[0, "Dst IP"] = "192.0.2.10"
    batch = validate_flow_records(frame)
    assert batch.frame.loc[0, "Src IP"] == "2001:db8::1"
    assert batch.frame.loc[0, "Dst IP"] == "192.0.2.10"

    invalid = valid_flow_frame.copy()
    invalid.loc[0, "Src IP"] = "192.0.2.999"
    with pytest.raises(InputContractError, match="valid IPv4/IPv6"):
        validate_flow_records(invalid)


@pytest.mark.parametrize(
    "timestamp",
    [
        "03/11/2024 01:30:00 AM",  # America/New_York fall-back fold
        "10/03/2024 02:30:00 AM",  # America/New_York spring-forward gap
    ],
)
def test_replay_rejects_ambiguous_and_nonexistent_source_local_times(
    tmp_path: Path,
    valid_flow_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    timestamp: str,
) -> None:
    model = _configure_fake_model(monkeypatch, tmp_path)
    directory = tmp_path / "rotations"
    directory.mkdir()
    frame = _flow_frame(valid_flow_frame, 1)
    frame.loc[0, "Timestamp"] = timestamp
    _write_rotation(directory, "01.csv", frame)
    with pytest.raises(InputContractError, match="Ambiguous or nonexistent"):
        _run(
            directory,
            tmp_path / "replay.sqlite",
            tmp_path / "incidents.jsonl",
            model=model,
            timezone="America/New_York",
        )


def test_real_model_replay_matches_sorted_whole_file_when_configured(
    tmp_path: Path,
) -> None:
    configured = os.environ.get("SECURITY_ANOMALY_E2E_MODEL")
    if not configured:
        pytest.skip("Set SECURITY_ANOMALY_E2E_MODEL for real-model replay parity")
    model = Path(configured).resolve()
    if not model.is_file():
        pytest.fail(f"SECURITY_ANOMALY_E2E_MODEL does not point to a file: {model}")

    frame = pd.read_csv(FIXTURE)
    source_timezone = "America/Los_Angeles"
    full_directory = tmp_path / "full-rotations"
    full_directory.mkdir()
    _write_rotation(full_directory, "all.csv", frame)
    full_output = tmp_path / "full.jsonl"
    replay.replay_directory(
        full_directory,
        model=model,
        sensor="e2e-sensor",
        timezone=source_timezone,
        checkpoint=tmp_path / "full.sqlite",
        output=full_output,
    )

    split_directory = tmp_path / "split-rotations"
    split_directory.mkdir()
    _write_rotation(split_directory, "02-late.csv", frame.iloc[6:].reset_index(drop=True))
    split_checkpoint = tmp_path / "split.sqlite"
    split_output = tmp_path / "split.jsonl"
    first = replay.replay_directory(
        split_directory,
        model=model,
        sensor="e2e-sensor",
        timezone=source_timezone,
        checkpoint=split_checkpoint,
        output=split_output,
    )
    assert first.rescored is True
    _write_rotation(split_directory, "01-early.csv", frame.iloc[:6].reset_index(drop=True))
    second = replay.replay_directory(
        split_directory,
        model=model,
        sensor="e2e-sensor",
        timezone=source_timezone,
        checkpoint=split_checkpoint,
        output=split_output,
    )
    assert second.rescored is True
    assert split_output.read_bytes() == full_output.read_bytes()

    # Use a fresh Python process for both restart paths.  The command must reuse the
    # snapshot without deserializing/scoring the model and then recover a missing file.
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), process_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-c",
        "from src.security_anomaly.cli import main; raise SystemExit(main())",
        "replay",
        str(split_directory),
        "--model",
        str(model),
        "--sensor",
        "e2e-sensor",
        "--timezone",
        source_timezone,
        "--checkpoint",
        str(split_checkpoint),
        "--output",
        str(split_output),
    ]
    unchanged_process = subprocess.run(
        command,
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
    )
    assert unchanged_process.returncode == 0, unchanged_process.stderr
    unchanged_summary = json.loads(unchanged_process.stderr.strip().splitlines()[-1])
    assert unchanged_summary["status"] == "unchanged"
    assert unchanged_summary["rescored"] is False
    expected_split = split_output.read_bytes()
    split_output.unlink()
    recovered_process = subprocess.run(
        command,
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
    )
    assert recovered_process.returncode == 0, recovered_process.stderr
    recovered_summary = json.loads(recovered_process.stderr.strip().splitlines()[-1])
    assert recovered_summary["status"] == "recovered"
    assert recovered_summary["rescored"] is False
    assert split_output.read_bytes() == expected_split

    from src.security_anomaly.detector import SecurityAnomalyDetector
    from src.security_anomaly.serialization import jsonl_text
    from src.security_anomaly.timeutils import source_utc, source_zone
    from src.security_anomaly.validation import (
        TIMESTAMP_FORMAT,
        read_and_validate_flow_csv,
    )

    detector = SecurityAnomalyDetector.load(model, MANIFEST, CONTRACT)
    expected_frame = frame.copy()
    zone = source_zone(source_timezone)
    expected_frame["Timestamp"] = [
        source_utc(
            pd.to_datetime(value, format=TIMESTAMP_FORMAT).to_pydatetime(), zone
        )
        .replace(tzinfo=None)
        .strftime(TIMESTAMP_FORMAT)
        for value in expected_frame["Timestamp"]
    ]
    expected_csv = tmp_path / "expected-utc.csv"
    expected_frame.to_csv(expected_csv, index=False)
    direct = detector.analyze_batch(read_and_validate_flow_csv(expected_csv).frame)
    expected = jsonl_text(promoted_incidents_to_v1(direct)).encode("utf-8")
    assert split_output.read_bytes() == expected
