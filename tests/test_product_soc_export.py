import json
import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.security_anomaly import cli, sentinel
from src.security_anomaly.ocsf import incident_finding
from src.security_anomaly.validation import InputContractError

INCIDENTS = Path(__file__).parent / "fixtures/product-v01/expected-incidents.jsonl"
RECORD = json.loads(INCIDENTS.read_text().splitlines()[0])


def test_sensor_identity_and_snapshot_identity_are_distinct():
    base = incident_finding(RECORD, "east", ZoneInfo("UTC"))
    updated = dict(RECORD, last_seen="2026-08-20T13:00:08")
    later = incident_finding(updated, "east", ZoneInfo("UTC"))
    other = incident_finding(RECORD, "west", ZoneInfo("UTC"))
    assert base["finding_info"]["uid"] == later["finding_info"]["uid"]
    assert base["metadata"]["uid"] != later["metadata"]["uid"]
    assert base["finding_info"]["uid"] != other["finding_info"]["uid"]
    assert base["type_uid"] == base["class_uid"] * 100 + base["activity_id"] == 200401
    evidence = base["evidences"][0]
    assert evidence["uid"] == RECORD["incident_id"]
    assert evidence["data"]["flow_count"] == 3
    assert evidence["data"]["max_attack_score"] == RECORD["max_attack_score"]
    assert evidence["dst_endpoint"] == {"ip": "198.51.100.20", "port": 445}
    assert (
        incident_finding(RECORD, "east", ZoneInfo("America/New_York"))["time"]
        == base["time"] + 4 * 3600_000
    )


@pytest.mark.parametrize("stamp", ["2026-11-01T01:30:00", "2026-03-08T02:30:00"])
def test_ambiguous_and_nonexistent_times_are_rejected(stamp):
    with pytest.raises(InputContractError, match="nonexistent"):
        incident_finding(
            dict(RECORD, first_seen=stamp, last_seen=stamp),
            "east",
            ZoneInfo("America/New_York"),
        )


def test_local_sentinel_request_preserves_evidence_and_enforces_byte_limit(
    tmp_path, monkeypatch
):
    ocsf = tmp_path / "findings.jsonl"
    batch = tmp_path / "request.json"
    assert (
        cli.main(
            [
                "export-ocsf",
                str(INCIDENTS),
                "--sensor",
                "edge",
                "--timezone",
                "UTC",
                "--output",
                str(ocsf),
            ]
        )
        == 0
    )
    args = ["sentinel-batch", str(ocsf), "--output", str(batch)]
    assert cli.main(args) == 0
    stored = batch.read_bytes()
    rows = json.loads(stored)
    assert [r["Finding"] for r in rows] == list(
        map(json.loads, ocsf.read_text().splitlines())
    )
    assert len(rows) == 2
    assert datetime.fromisoformat(rows[0]["TimeGenerated"]) == datetime(
        2026, 8, 20, 13, 0, 5, tzinfo=UTC
    )
    monkeypatch.setattr(sentinel, "MAX_BATCH_BYTES", len(stored) - 1)
    assert cli.main(args) == cli.EXIT_INPUT
    assert batch.read_bytes() == stored


def test_invalid_second_incident_cannot_replace_a_good_export(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_text(
        json.dumps(RECORD) + "\n" + json.dumps(dict(RECORD, promoted=False))
    )
    output.write_bytes(b"previous")
    args = [
        "export-ocsf",
        str(source),
        "--sensor",
        "edge",
        "--timezone",
        "UTC",
        "--output",
        str(output),
    ]
    assert cli.main(args) == cli.EXIT_INPUT
    assert output.read_bytes() == b"previous"
    alias = tmp_path / "alias"
    os.link(source, alias)
    assert cli.main([*args[:-1], str(alias)]) == cli.EXIT_OUTPUT


def test_empty_capture_is_an_empty_local_batch(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.touch()
    assert cli.main(["sentinel-batch", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == []
