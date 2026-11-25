from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.security_anomaly import zeek

FIXTURES = Path(__file__).parent / "fixtures" / "zeek"


def _fake_finding(**kwargs: object) -> dict[str, object]:
    """Small OCSF-shaped stand-in; the real builder is tested separately."""

    return {
        "class_uid": 2004,
        "sensor": kwargs["sensor"],
        "observation_id": kwargs["observation_id"],
        "time": kwargs["first_seen"].isoformat(),  # type: ignore[union-attr]
        "src_endpoint": {
            "ip": kwargs["src_ip"],
            "port": kwargs["src_port"],
        },
        "dst_endpoint": {
            "ip": kwargs["dst_ip"],
            "port": kwargs["dst_port"],
        },
        "detector": kwargs["detector"],
        "detector_version": kwargs["detector_version"],
        "title": kwargs["title"],
        "evidence": kwargs["evidence"],
        "product_version": kwargs["product_version"],
    }


def test_parser_counts_invalid_incomplete_duplicate_and_uses_utc() -> None:
    batch = zeek.read_zeek_jsonl(FIXTURES / "heldout.jsonl")

    assert batch.stats.lines_read == 7
    assert batch.stats.parsed_records == 4
    assert batch.stats.scorable_records == 3
    assert batch.stats.excluded_records == 3
    assert batch.stats.incomplete_records == 1
    assert batch.stats.invalid_records == 2
    assert batch.stats.duplicate_records == 1
    assert batch.records[0].timestamp.isoformat() == "2025-01-01T00:01:40+00:00"
    assert batch.records[-1].src_ip == "2001:db8::50"
    assert batch.records[-1].protocol == "tcp"


@pytest.mark.parametrize(
    ("field", "value"),
    [("id.orig_p", 42000.0), ("orig_pkts", 4.5), ("missed_bytes", -1)],
)
def test_parser_rejects_non_integral_ports_and_invalid_counters(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads((FIXTURES / "fit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    payload[field] = value
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    batch = zeek.read_zeek_jsonl(source)
    assert batch.stats.invalid_records == 1
    assert batch.stats.parsed_records == 0


def test_parser_counts_missing_identity_as_incomplete(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "fit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    del payload["id.resp_p"]
    source = tmp_path / "incomplete.jsonl"
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    batch = zeek.read_zeek_jsonl(source)
    assert batch.stats.incomplete_records == 1
    assert batch.stats.invalid_records == 0
    assert batch.stats.excluded_records == 1


def test_fit_is_separate_and_score_marks_only_held_out_unusual_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zeek, "detection_finding", _fake_finding)
    profile = zeek.fit_profile(
        zeek.read_zeek_jsonl(FIXTURES / "fit.jsonl"),
        sensor="lab-sensor-a",
        threshold=6.0,
    )

    assert profile.fit_records == 8
    assert profile.sensor == "lab-sensor-a"
    assert profile.feature_names == zeek.PROFILE_FEATURES
    assert profile.center["duration"] == pytest.approx(0.2)

    result = zeek.score_batch(
        zeek.read_zeek_jsonl(FIXTURES / "heldout.jsonl"),
        profile,
        sensor="lab-sensor-a",
    )
    assert result.parsed_records == 4
    assert result.scored_records == 3
    assert result.anomalous_records == 1
    assert result.excluded_records == 3
    assert result.incomplete_records == 1
    finding = result.findings[0]
    assert finding["class_uid"] == 2004
    assert finding["sensor"] == "lab-sensor-a"
    assert finding["observation_id"] == "H2"
    assert finding["src_endpoint"]["ip"] == "192.0.2.51"
    assert finding["dst_endpoint"]["port"] == 8443
    assert finding["evidence"]["imputation"] == "none"
    assert finding["evidence"]["measured"]["orig_bytes"] == 9000.0
    assert finding["evidence"]["training_baseline"]["fit_records"] == 8
    assert finding["evidence"]["score"] >= 6.0


def test_score_uses_shared_ocsf_detection_finding_builder() -> None:
    profile = zeek.fit_profile(
        zeek.read_zeek_jsonl(FIXTURES / "fit.jsonl"),
        sensor="lab-sensor-a",
    )
    result = zeek.score_batch(
        zeek.read_zeek_jsonl(FIXTURES / "heldout.jsonl"),
        profile,
        sensor="lab-sensor-a",
    )

    finding = result.findings[0]
    assert finding["class_uid"] == 2004
    assert finding["metadata"]["log_provider"] == "lab-sensor-a"
    assert finding["finding_info"]["analytic"] == {
        "name": "zeek-profile",
        "version": "zeek-profile-v1",
        "type_id": 3,
    }
    assert finding["evidences"][0]["uid"] == "H2"
    assert finding["end_time"] - finding["start_time"] == 30_000
    evidence = finding["evidences"][0]["data"]
    assert evidence["profile_fingerprint"] == profile.fingerprint
    assert evidence["training_baseline"]["profile_fingerprint"] == profile.fingerprint
    assert evidence["imputation"] == "none"


def test_register_parser_exposes_handler_and_run_and_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(zeek, "detection_finding", _fake_finding)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    zeek.register_parser(subparsers)

    profile_path = tmp_path / "profile.json"
    fit_args = parser.parse_args(
        [
            "zeek-fit",
            str(FIXTURES / "fit.jsonl"),
            "--sensor",
            "lab-sensor-a",
            "--output",
            str(profile_path),
        ]
    )
    assert fit_args.handler(fit_args) == 0
    assert json.loads(profile_path.read_text(encoding="utf-8"))["sensor"] == "lab-sensor-a"
    assert "zeek-fit" in capsys.readouterr().err

    output_path = tmp_path / "findings.jsonl"
    score_args = parser.parse_args(
        [
            "zeek-score",
            str(FIXTURES / "heldout.jsonl"),
            "--profile",
            str(profile_path),
            "--sensor",
            "lab-sensor-a",
            "--output",
            str(output_path),
        ]
    )
    assert score_args.handler(score_args) == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["observation_id"] for row in rows] == ["H2"]
    assert '"incomplete_records":1' in capsys.readouterr().err


def test_profile_cannot_be_reused_for_another_sensor() -> None:
    profile = zeek.fit_profile(
        zeek.read_zeek_jsonl(FIXTURES / "fit.jsonl"),
        sensor="lab-sensor-a",
    )
    with pytest.raises(zeek.ZeekProfileError, match="does not match"):
        zeek.score_batch(
            zeek.read_zeek_jsonl(FIXTURES / "heldout.jsonl"),
            profile,
            sensor="other-sensor",
        )


def test_unscorable_count_counts_each_input_observation_once(tmp_path):
    source = tmp_path / "missing-fields.jsonl"
    example = json.loads((FIXTURES / "heldout.jsonl").read_text().splitlines()[0])
    incomplete = {k: v for k, v in example.items() if k != "orig_pkts"}
    rows = [{}, incomplete, incomplete, dict(example, uid="complete")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    stats = zeek.read_zeek_jsonl(source).stats
    assert stats.scorable_records == 1
    assert stats.not_scorable_records == 3
