from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.security_anomaly import cli, evaluation, review, zeek
from src.security_anomaly.validation import InputContractError


FIXTURES = Path(__file__).parent / "fixtures" / "periodic"


def _batches() -> tuple[zeek.ZeekBatch, zeek.ZeekBatch]:
    return (
        zeek.read_zeek_jsonl(FIXTURES / "reference.jsonl"),
        zeek.read_zeek_jsonl(FIXTURES / "observed.jsonl"),
    )


def test_review_queue_is_unlabeled_deterministic_and_budgeted() -> None:
    reference, observed = _batches()
    first, coverage = review.build_queue(
        reference,
        observed,
        ["192.0.2.0/24"],
        "edge-a",
        method="rare_periodic",
        budget=2,
    )
    second, same_coverage = review.build_queue(
        reference,
        observed,
        ["192.0.2.0/24"],
        "edge-a",
        method="rare_periodic",
        budget=2,
    )
    assert first == second
    assert coverage == same_coverage
    assert len(first) == 2
    assert [item["rank"] for item in first] == [1, 2]
    assert coverage["eligible_case_count"] == coverage["available_case_count"]
    assert coverage["selected_case_count"] == 2
    assert all(item["uids"] for item in first)
    assert all(item["snapshot_id"].startswith("snap_") for item in first)
    assert all(item["diagnostic"]["evidence"]["observed_uids"] == item["uids"] for item in first)


def test_snapshot_is_independent_of_rank_and_budget() -> None:
    reference, observed = _batches()
    small, _ = review.build_queue(reference, observed, ["192.0.2.0/24"], "edge-a", budget=1)
    large, _ = review.build_queue(reference, observed, ["192.0.2.0/24"], "edge-a", budget=4)
    assert small[0]["case_id"] == large[0]["case_id"]
    assert small[0]["snapshot_id"] == large[0]["snapshot_id"]
    assert small[0]["rank"] == 1
    assert large[0]["rank"] == 1


def test_review_methods_have_their_own_scores_and_threshold_decisions() -> None:
    reference, observed = _batches()
    periodic_queue, periodic_coverage = review.build_queue(
        reference, observed, ["192.0.2.0/24"], "edge-a", method="periodicity_only", budget=20
    )
    mad_queue, mad_coverage = review.build_queue(
        reference, observed, ["192.0.2.0/24"], "edge-a", method="zeek_mad", budget=20
    )
    assert periodic_coverage["available_case_count"] == len(periodic_queue)
    assert mad_coverage["available_case_count"] == len(mad_queue)
    assert all(item["method"] == "periodicity_only" for item in periodic_queue)
    assert all(item["threshold"] == pytest.approx(0.8) for item in periodic_queue)
    assert all(item["method"] == "zeek_mad" for item in mad_queue)
    assert all("maximum_member" in item["diagnostic"]["evidence"] for item in mad_queue)


def test_zero_and_unavailable_scores_are_reported_separately() -> None:
    reference, observed = _batches()
    queue, coverage = review.build_queue(
        reference, observed, ["192.0.2.0/24"], "edge-a", method="zeek_mad", budget=0
    )
    assert queue == []
    assert coverage["selected_case_count"] == 0
    assert coverage["available_case_count"] >= coverage["zero_score_case_count"]
    assert coverage["unavailable_case_count"] == 0


def test_snapshot_fingerprint_changes_with_membership() -> None:
    evidence = {"observed_uids": ["u1"], "score": 2.0}
    first = review.snapshot_fingerprint(
        {"sensor": "edge-a", "method": "zeek_mad", "case_id": "case", "uids": ["u1"], "diagnostic": evidence}
    )
    second = review.snapshot_fingerprint(
        {"sensor": "edge-a", "method": "zeek_mad", "case_id": "case", "uids": ["u1", "u2"], "diagnostic": evidence}
    )
    assert first != second


def test_changed_nonwinning_record_keeps_case_but_requires_a_new_review() -> None:
    reference, observed = _batches()
    def change(batch, uid, feature, value):
        return replace(batch, records=tuple(
            replace(r, features={**r.features, feature: value}) if r.uid == uid else r for r in batch.records
        ))
    observed = change(observed, "update-3", "resp_pkts", 10000)
    corrected = change(observed, "update-0", "resp_bytes", 1)
    def row(batch):
        items, _ = review.build_queue(reference, batch, ["192.0.2.0/24"], "edge", budget=100)
        return next(item for item in items if "update-3" in item["uids"])
    before, after = row(observed), row(corrected)
    assert before["case_id"] == after["case_id"] and before["score"] == after["score"]
    assert before["snapshot_id"] != after["snapshot_id"]


@pytest.mark.parametrize("method", review.METHODS)
def test_queue_and_evaluation_match_on_the_same_method_pool(method):
    reference, observed = _batches()
    nets, sensor = ["192.0.2.0/24"], "edge"
    cases, _ = evaluation._build_review_cases(reference, observed, nets, sensor)
    labels = {r.uid: {"label": "benign"} for r in observed.records}
    compared = evaluation.compare_cases(cases, labels, budget=2, methods=[method])
    rows, coverage = review.build_queue(reference, observed, nets, sensor, method=method, budget=2)
    assert coverage["available_case_count"] == compared["common_pool_count"]
    assert [r["case_id"] for r in rows] == compared["methods"][method]["selected_case_ids"]


def test_cli_review_writes_jsonl_without_labels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "queue.jsonl"
    code = cli.main(
        [
            "review",
            "zeek",
            str(FIXTURES / "observed.jsonl"),
            "--reference",
            str(FIXTURES / "reference.jsonl"),
            "--sensor",
            "edge-a",
            "--local-net",
            "192.0.2.0/24",
            "--method",
            "rare_periodic",
            "--budget",
            "1",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["rank"] == 1
    assert '"command":"review zeek"' in capsys.readouterr().err


@pytest.mark.parametrize("method", ["bad", "random"])
def test_review_rejects_unknown_method(method: str) -> None:
    reference, observed = _batches()
    with pytest.raises(InputContractError, match="method"):
        review.build_queue(reference, observed, ["192.0.2.0/24"], "edge-a", method=method)
