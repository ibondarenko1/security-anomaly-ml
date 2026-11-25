import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.security_anomaly import cli
from src.security_anomaly.cic_evaluation import _observation_ids
from src.security_anomaly.validation import InputContractError


FLOWS = Path(__file__).parent / "fixtures" / "product-v01" / "flows.csv"


def test_row_ids_follow_the_exact_input_and_flow_ids_remain_text():
    raw = b'Flow ID,extra\n001,a\n"NA",b\n'
    digest = hashlib.sha256(raw).hexdigest()
    assert _observation_ids(raw, digest, 2, "flow-id") == ("001", "NA")
    assert _observation_ids(raw, digest, 2, "row") == (f"{digest}:1", f"{digest}:2")
    with pytest.raises(InputContractError, match="repeat"):
        _observation_ids(raw.replace(b'"NA"', b"001"), digest, 2, "flow-id")


def test_cic_real_model_compares_identical_queues_and_separate_operating_point(tmp_path):
    model = os.environ.get("SECURITY_ANOMALY_E2E_MODEL")
    if not model:
        pytest.skip("set SECURITY_ANOMALY_E2E_MODEL to test the frozen RF")
    with FLOWS.open(encoding="utf-8", newline="") as handle:
        ids = [r["Flow ID"] for r in csv.DictReader(handle)]
    rows = [
        {"uid": uid, "label": "target", "episode": "one-authored-episode"}
        if uid.endswith("-bulk") else {"uid": uid, "label": "benign"}
        for uid in ids
    ]
    sidecar = tmp_path / "truth.jsonl"
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "comparison.json"
    args = ["evaluate", "cic", str(FLOWS), "--model", model, "--budget", "2",
            "--labels", str(sidecar), "--output", str(output)]
    assert cli.main(args) == 0
    snapshot = output.read_bytes()
    report = json.loads(snapshot)
    assert report["coverage"] == {
        "accepted_flows": 12, "all_flow_groups": 7, "alert_flows": 9, "alert_groups": 5,
        "promoted_groups": 2,
    }
    methods = report["comparison"]["methods"]
    assert set(methods) == {"frozen_rf", "packet_count", "random"}
    assert all(m["selected_count"] == 2 for m in methods.values())
    assert methods["frozen_rf"]["selected_target_cases"] == 2
    assert methods["frozen_rf"]["selected_target_episodes"] == 1
    assert report["operating_point"]["methods"]["frozen_rf"]["selected_count"] == 2
    assert report["input_sha256"] == hashlib.sha256(FLOWS.read_bytes()).hexdigest()
    # A missing label fails the run and leaves the previous report intact.
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
    assert cli.main(args) == cli.EXIT_INPUT
    assert output.read_bytes() == snapshot
