from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.security_anomaly import periodic, zeek


def _row(
    uid: str,
    timestamp: float,
    *,
    source: str = "10.0.0.2",
    destination: str = "8.8.8.8",
    protocol: str = "tcp",
    destination_port: int = 443,
    source_port: int = 40000,
    duration: float = 0.2,
) -> dict[str, object]:
    return {
        "ts": timestamp,
        "uid": uid,
        "id.orig_h": source,
        "id.orig_p": source_port,
        "id.resp_h": destination,
        "id.resp_p": destination_port,
        "proto": protocol,
        "duration": duration,
        "orig_bytes": 10,
        "resp_bytes": 20,
        "orig_pkts": 2,
        "resp_pkts": 3,
        "orig_ip_bytes": 30,
        "resp_ip_bytes": 40,
        "missed_bytes": 0,
    }


def _batch(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> zeek.ZeekBatch:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return zeek.read_zeek_jsonl(path)


def test_build_cases_groups_sessions_and_scores_regular_rare_destination(tmp_path: Path) -> None:
    reference = _batch(
        tmp_path,
        "reference",
        [
            _row("r1", 1, destination="8.8.8.8"),
            _row("r2", 2, destination="8.8.8.8"),
            _row("r3", 3, destination="1.1.1.1"),
        ],
    )
    observed = _batch(
        tmp_path,
        "observed",
        [
            _row("o1", 1000, destination="9.9.9.9"),
            _row("o2", 1100, destination="9.9.9.9"),
            _row("o3", 1200, destination="9.9.9.9"),
            _row("o4", 1300, destination="9.9.9.9"),
        ],
    )

    cases = periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")

    assert len(cases) == 1
    case = cases[0]
    assert case.case_id.startswith("pc_")
    assert [record.uid for record in case.records] == ["o1", "o2", "o3", "o4"]
    assert case.distinct_start_count == 4
    assert case.interval_median_seconds == pytest.approx(100.0)
    assert case.interval_mean_absolute_error_seconds == pytest.approx(0.0)
    assert case.periodicity == pytest.approx(1.0)
    assert case.reference_origin_count == 3
    assert case.reference_destination_count == 0
    assert case.rarity == pytest.approx(1.0)
    assert case.score == pytest.approx(1.0)
    assert case.flagged


def test_all_eligible_groups_include_singletons_and_gap_splits(tmp_path: Path) -> None:
    reference = _batch(
        tmp_path,
        "reference",
        [_row("r1", 1), _row("r2", 2), _row("r3", 3)],
    )
    observed = _batch(
        tmp_path,
        "observed",
        [
            _row("a1", 1000, destination="9.9.9.9"),
            _row("a2", 1200, destination="9.9.9.9"),
            _row("b1", 3101, destination="9.9.9.9"),
            _row("c1", 1000, destination="7.7.7.7", protocol="udp", destination_port=53),
        ],
    )

    cases = periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")

    assert [(case.destination_ip, [row.uid for row in case.records]) for case in cases] == [
        ("7.7.7.7", ["c1"]),
        ("9.9.9.9", ["a1", "a2"]),
        ("9.9.9.9", ["b1"]),
    ]
    assert all(case.periodicity == 0 for case in cases)
    assert all(not case.flagged for case in cases)


@pytest.mark.parametrize(
    "destination",
    ["10.0.0.9", "127.0.0.1", "0.0.0.0", "224.0.0.1", "255.255.255.255",
     "169.254.1.5", "fe80::1", "ff02::1"],
)
def test_destination_filters_and_direction_are_fail_closed(
    tmp_path: Path, destination: str
) -> None:
    reference = _batch(tmp_path, "reference", [_row("r1", 1), _row("r2", 2), _row("r3", 3)])
    observed = _batch(
        tmp_path,
        "observed",
        [
            _row("o1", 1000, destination=destination),
            _row("o2", 1100, destination=destination),
            _row("o3", 1200, destination=destination),
            _row("o4", 1300, destination=destination),
            _row("inbound", 1400, source="8.8.8.8", destination="10.0.0.2"),
        ],
    )

    assert periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a") == []


def test_build_cases_accepts_incomplete_feature_records_for_timing(tmp_path: Path) -> None:
    reference = _batch(tmp_path, "reference", [_row("r1", 1), _row("r2", 2), _row("r3", 3)])
    incomplete = _row("o2", 1100, destination="9.9.9.9")
    del incomplete["orig_bytes"]
    observed = _batch(
        tmp_path,
        "observed",
        [
            _row("o1", 1000, destination="9.9.9.9"),
            incomplete,
            _row("o3", 1200, destination="9.9.9.9"),
            _row("o4", 1300, destination="9.9.9.9"),
        ],
    )

    case = periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")[0]

    assert observed.stats.incomplete_records == 1
    assert [record.uid for record in case.records] == ["o1", "o2", "o3", "o4"]
    assert case.periodicity == pytest.approx(1.0)


def test_strict_time_uid_network_and_sensor_validation(tmp_path: Path) -> None:
    reference = _batch(tmp_path, "reference", [_row("r1", 1), _row("r2", 2), _row("r3", 3)])
    observed = _batch(tmp_path, "observed", [_row("o1", 3)])
    with pytest.raises(periodic.PeriodicInputError, match="strictly earlier"):
        periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")

    observed = _batch(tmp_path, "overlap", [_row("r1", 1000)])
    with pytest.raises(periodic.PeriodicInputError, match="disjoint"):
        periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")

    observed = _batch(tmp_path, "valid", [_row("o1", 1000)])
    with pytest.raises(periodic.PeriodicInputError, match="CIDR"):
        periodic.build_cases(reference, observed, [], "edge-a")
    with pytest.raises(periodic.PeriodicInputError):
        periodic.build_cases(reference, observed, ["invalid"], "edge-a")
    with pytest.raises(periodic.PeriodicInputError):
        periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge sensor")


def test_findings_are_stable_and_include_original_uids(tmp_path: Path) -> None:
    reference = _batch(
        tmp_path,
        "reference",
        [_row("r1", 1), _row("r2", 2), _row("r3", 3)],
    )
    observed = _batch(
        tmp_path,
        "observed",
        [_row(f"o{i}", 1000 + i * 100, destination="9.9.9.9") for i in range(4)],
    )
    cases = periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a")

    first = periodic.findings_for_cases(cases, "edge-a")
    second = periodic.findings_for_cases(periodic.build_cases(reference, observed, ["10.0.0.0/24"], "edge-a"), "edge-a")

    assert first == second
    assert len(first) == 1
    finding = first[0]
    assert finding["class_uid"] == 2004
    assert finding["finding_info"]["analytic"]["name"] == periodic.DETECTOR_NAME
    assert finding["evidences"][0]["uid"] == cases[0].case_id
    assert finding["evidences"][0]["data"]["observed_uids"] == ["o0", "o1", "o2", "o3"]
    assert finding["evidences"][0]["data"]["reference_origin_count"] == 3
    assert finding["evidences"][0]["data"]["intervals_seconds"] == [100.0, 100.0, 100.0]
    assert finding["evidences"][0]["data"]["first_seen"].endswith("+00:00")
    assert finding["evidences"][0]["data"]["last_seen"].endswith("+00:00")
    assert finding["finding_info"]["uid"] == second[0]["finding_info"]["uid"]


def test_case_id_is_logical_session_identity_while_snapshot_uid_changes(
    tmp_path: Path,
) -> None:
    reference = _batch(
        tmp_path,
        "reference",
        [_row("r1", 1), _row("r2", 2), _row("r3", 3)],
    )
    first_observed = _batch(
        tmp_path,
        "first-observed",
        [_row(f"o{i}", 1000 + i * 100, destination="9.9.9.9") for i in range(4)],
    )
    extended_observed = _batch(
        tmp_path,
        "extended-observed",
        [_row(f"o{i}", 1000 + i * 100, destination="9.9.9.9") for i in range(5)],
    )

    first_case = periodic.build_cases(reference, first_observed, ["10.0.0.0/24"], "edge-a")[0]
    extended_case = periodic.build_cases(
        reference, extended_observed, ["10.0.0.0/24"], "edge-a"
    )[0]
    first_finding = periodic.findings_for_cases((first_case,), "edge-a")[0]
    extended_finding = periodic.findings_for_cases((extended_case,), "edge-a")[0]

    assert first_case.case_id == extended_case.case_id
    assert first_finding["finding_info"]["uid"] == extended_finding["finding_info"]["uid"]
    assert first_finding["metadata"]["uid"] != extended_finding["metadata"]["uid"]


def test_register_parser_writes_only_flagged_ocsf_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_path = tmp_path / "reference.jsonl"
    observed_path = tmp_path / "observed.jsonl"
    output_path = tmp_path / "findings.jsonl"
    reference_path.write_text(
        "".join(json.dumps(_row(f"r{i}", i + 1)) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    observed_path.write_text(
        "".join(
            json.dumps(_row(f"o{i}", 1000 + i * 100, destination="9.9.9.9")) + "\n"
            for i in range(4)
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    periodic.register_parser(subparsers)
    args = parser.parse_args(
        [
            "zeek-periodic",
            str(observed_path),
            "--reference",
            str(reference_path),
            "--local-net",
            "10.0.0.0/24",
            "--sensor",
            "edge-a",
            "--output",
            str(output_path),
        ]
    )

    assert args.handler(args) == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["class_uid"] == 2004
    assert '"flagged_sessions":1' in capsys.readouterr().err
