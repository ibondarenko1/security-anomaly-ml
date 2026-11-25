from __future__ import annotations

import json
import sys

import pytest

from tools import evaluate_iot23_episodes as replay


def labelled_stream():
    # The publisher packs the last core column and both labels into one tab field.
    header = [
        "#fields\tts\tuid\tid.orig_h\tid.resp_h\tid.orig_p\tid.resp_p\tproto\ttunnel_parents   label   detailed-label",
        "#types\ttime\tstring\taddr\taddr\tport\tport\tenum\tset[string]   string   string",
    ]
    times = [1, 2, 3, 4, 5, 6, 7, 7, 4000, 4001]
    lines = list(header)
    for i, ts in enumerate(times):
        label = "Malicious   C&C-HeartBeat" if i in (1, 7, 8) else "Benign   -"
        lines.append(f"{ts}\tu{i}\t192.0.2.1\t198.51.100.4\t10000\t443\ttcp\t-   {label}")
    return ("\n".join(lines) + "\n").encode()


def test_publisher_sidecar_split_and_new_episodes():
    source = labelled_stream()
    rows, labels = replay.parse(source)
    assert all("label" not in r and "detailed-label" not in r for r in rows)
    tabbed = source.replace(b"   ", b"\t").replace(b"detailed-label", b"det_label")
    assert replay.parse(tabbed) == (rows, labels)
    assert labels["u7"] == ("target", "c&c-heartbeat")
    reference, holdout = replay.split(rows)
    assert len(reference) == 6
    assert reference[-1]["ts"] < holdout[0]["ts"]
    episodes = replay.assign_episodes(rows, labels, "sensor", 3600)
    assert episodes["u1"] == episodes["u7"] != episodes["u8"]
    with pytest.raises(ValueError, match="duplicate"):
        replay.parse(source.replace(b"\tu7\t", b"\tu0\t"))
    with pytest.raises(ValueError, match="unexpected publisher label"):
        replay.parse(source.replace(b"Benign ", b"Unknown "))


def test_benchmark_cli_leaves_only_reproducible_public_counts(tmp_path, monkeypatch):
    scenario = {"id": "example", "source_sha256": "fixture", "local_networks": ["192.0.2.0/24"]}
    protocol = {
        "protocol": "fixture", "scenarios": [scenario],
        "input_contract": {"max_source_bytes": 8000, "budget": 10, "seed": "fixture", "episode_gap_seconds": 3600},
    }
    manifest = tmp_path / "protocol.json"
    manifest.write_text(json.dumps(protocol), encoding="utf8")
    monkeypatch.setattr(replay, "PROTOCOL_PATH", manifest)
    monkeypatch.setattr(replay, "download", lambda spec, limit: labelled_stream())
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["replay", "--output-dir", str(output), "--evaluate"])
    replay.main()
    aggregate = output / "aggregate.json"
    assert list(output.iterdir()) == [aggregate]
    raw = aggregate.read_bytes()
    assert b"\r" not in raw and b"selected_episode_ids" not in raw and b"reviewed_cases" not in raw
    row, = json.loads(raw)["captures"]
    assert row["holdout"]["target_episodes"] == 2
    assert row["new_holdout_episodes"] == row["spillover_episodes"] == 1
    picked = row["comparison"]["methods"]["random"]
    assert picked["selected_target_episodes"] == 2
    assert picked["selected_new_episodes"] == picked["selected_spillover_episodes"] == 1
