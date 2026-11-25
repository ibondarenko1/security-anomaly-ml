from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.security_anomaly import evaluation
from src.security_anomaly.validation import InputContractError


def case(case_id: str, score: float, uids: tuple[str, ...] = ()) -> evaluation.ReviewCase:
    return evaluation.ReviewCase(
        case_id=case_id,
        uids=uids or (f"{case_id}-uid",),
        scores={"rare_periodic": score, "periodicity_only": score, "zeek_mad": score},
    )


def labels(cases: list[evaluation.ReviewCase], target: set[str]) -> dict[str, dict[str, str]]:
    return {
        uid: ({"label": "target", "episode": f"episode-{case.case_id}"} if case.case_id in target else {"label": "benign"})
        for case in cases
        for uid in case.uids
    }


def test_compare_uses_one_pool_and_reproducible_hash_ties() -> None:
    cases = [case("a", 1.0), case("b", 0.9), case("c", 0.1), case("d", 0.0)]
    result = evaluation.compare_cases(cases, labels(cases, {"a", "c"}), 2, seed="fixed")
    assert result == evaluation.compare_cases(cases, labels(cases, {"a", "c"}), 2, seed="fixed")
    assert result["observation_count"] == 4
    assert result["methods"]["rare_periodic"]["selected_case_ids"] == ["a", "b"]
    assert result["methods"]["rare_periodic"]["selected_target_cases"] == 1
    assert result["methods"]["rare_periodic"]["total_target_cases"] == 2
    assert result["methods"]["rare_periodic"]["expected_target_cases_under_uniform"] == 1.0
    assert result["methods"]["rare_periodic"]["target_case_recall"] == 0.5


def test_target_episode_is_counted_once_for_mixed_cases() -> None:
    cases = [
        evaluation.ReviewCase("one", ("u1",), {"rare_periodic": 0.8}),
        evaluation.ReviewCase("two", ("u2",), {"rare_periodic": 0.7}),
    ]
    truth = {"u1": {"label": "target", "episode": "same"}, "u2": {"label": "target", "episode": "same"}}
    result = evaluation.compare_cases(cases, truth, 2)
    summary = result["methods"]["rare_periodic"]
    assert summary["total_target_cases"] == 2
    assert summary["total_target_episodes"] == 1
    assert summary["selected_target_episodes"] == 1
    assert summary["target_episode_recall"] == 1.0


def test_uniform_episode_expectation_and_out_of_scope_denominator() -> None:
    cases = [case(k, 0.1) for k in ("a", "b", "c", "d")]
    truth = labels(cases, set())
    for uid in ("a-uid", "b-uid"):
        truth[uid] = {"label": "target", "episode": "shared"}
    truth["inbound-only"] = {"label": "target", "episode": "outside-pool"}
    result = evaluation.compare_cases(cases, truth, 2)
    assert result["methods"]["rare_periodic"]["target_episode_recall"] == 0.5
    assert result["uniform_expectation"]["target_cases"] == 1.0
    assert result["uniform_expectation"]["distinct_target_episodes"] == pytest.approx(5 / 6)


def test_read_labels_rejects_missing_extra_duplicate_and_bad_episode(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"uid": "u1", "label": "target", "episode": "e1"}) + "\n", encoding="utf-8")
    with pytest.raises(InputContractError, match="missing"):
        evaluation.read_labels(path, {"u1", "u2"})
    path.write_text(json.dumps({"uid": "u1", "label": "target"}) + "\n", encoding="utf-8")
    with pytest.raises(InputContractError, match="episode"):
        evaluation.read_labels(path, {"u1"})
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"uid": "u1", "label": "benign"},
                {"uid": "u1", "label": "benign"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError, match="duplicate"):
        evaluation.read_labels(path, {"u1"})
    path.write_text(json.dumps({"uid": "extra", "label": "benign"}) + "\n", encoding="utf-8")
    with pytest.raises(InputContractError, match="unknown"):
        evaluation.read_labels(path, {"u1"})


def test_read_labels_allows_an_empty_capture_with_empty_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "empty-labels.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert evaluation.read_labels(path, set()) == {}
    path.write_text(json.dumps({"uid": "stale", "label": "benign"}) + "\n", encoding="utf-8")
    with pytest.raises(InputContractError, match="empty"):
        evaluation.read_labels(path, set())


def test_operating_point_uses_explicit_policy_decisions() -> None:
    cases = [
        evaluation.ReviewCase(
            "flagged",
            ("u1",),
            {"rare_periodic": 0.1},
            {"rare_periodic": True},
        ),
        evaluation.ReviewCase(
            "quiet",
            ("u2",),
            {"rare_periodic": 0.9},
            {"rare_periodic": False},
        ),
    ]
    result = evaluation.operating_point(cases, {"u1": {"label": "benign"}, "u2": {"label": "target", "episode": "e"}})
    assert result["methods"]["rare_periodic"]["selected_case_ids"] == ["flagged"]
    assert result["methods"]["rare_periodic"]["selected_pure_benign_cases"] == 1
    assert result["methods"]["rare_periodic"]["selected_target_cases"] == 0


def test_explicit_method_subset_and_empty_pool_are_valid() -> None:
    cases = [case("a", 1.0)]
    truth = labels(cases, {"a"})
    result = evaluation.compare_cases(cases, truth, 4, methods=("frozen_rf", "random"))
    assert tuple(result["methods"]) == ("frozen_rf", "random")
    assert result["budget"] == 1
    unavailable = evaluation.operating_point(cases, truth, methods=("frozen_rf",))["methods"]["frozen_rf"]
    assert unavailable["score_status"] == "unavailable"
    assert unavailable["target_episode_recall"] is None
    assert result["methods"]["frozen_rf"]["selected_count"] == 0
    assert result["methods"]["frozen_rf"]["target_episode_recall"] is None
    empty = evaluation.compare_cases([], {}, 4, methods=("frozen_rf",))
    assert empty["case_count"] == 0
    assert empty["budget"] == 0
    assert empty["methods"]["frozen_rf"]["target_case_recall"] is None


def test_mad_threshold_uses_inclusive_profile_decision_and_reports_fills() -> None:
    cases = [
        evaluation.ReviewCase(
            "at-threshold",
            ("u1",),
            {"zeek_mad": 6.0},
            {"zeek_mad": True},
        ),
        evaluation.ReviewCase("missing", ("u2",), {"zeek_mad": None}),
        evaluation.ReviewCase("zero", ("u3",), {"zeek_mad": 0.0}),
    ]
    truth = {uid: {"label": "benign"} for uid in ("u1", "u2", "u3")}
    point = evaluation.operating_point(cases, truth, methods=("zeek_mad",))
    assert point["methods"]["zeek_mad"]["selected_case_ids"] == ["at-threshold"]
    ranked = evaluation.compare_cases(cases, truth, 3, methods=("zeek_mad",))
    summary = ranked["methods"]["zeek_mad"]
    assert summary["score_status"] == "partial"
    assert summary["unavailable_case_count"] == 1
    assert ranked["common_pool_count"] == 2
    assert ranked["budget"] == 2
    assert summary["selected_unavailable_fills"] == 0
    assert summary["selected_zero_score_fills"] == 1


def test_zeek_uses_the_actual_profile_threshold_and_joins_uids(tmp_path: Path, monkeypatch) -> None:
    from src.security_anomaly import cli, zeek

    directory = Path(__file__).parent / "fixtures" / "periodic"
    scored = lambda record, profile: (6.0 if record.uid.startswith("callback-") else 5.99, {})
    monkeypatch.setattr(zeek, "_score_value", scored)
    output = tmp_path / "evaluation.json"
    assert cli.main([
        "evaluate", "zeek", str(directory / "observed.jsonl"), "--reference", str(directory / "reference.jsonl"),
        "--labels", str(directory / "labels.jsonl"), "--sensor", "edge-a", "--local-net", "192.0.2.0/24",
        "--budget", "1", "--output", str(output)
    ]) == 0
    report = json.loads(output.read_bytes())
    mad = report["operating_point"]["methods"]["zeek_mad"]
    assert mad["selected_count"] == mad["selected_target_cases"] == 1
    card = report["comparison"]["reviewed_cases"][mad["selected_case_ids"][0]]
    assert card["uids"] == [f"callback-{n}" for n in range(4)]

def test_changing_truth_cannot_change_queue_order_or_candidates() -> None:
    cases = [case("a", 0), case("b", 0), case("c", 1)]
    first = evaluation.compare_cases(cases, labels(cases, {"a"}), 2)
    second = evaluation.compare_cases(list(reversed(cases)), labels(cases, {"b", "c"}), 2)
    assert first["common_pool_count"] == second["common_pool_count"] == 3
    for name in first["methods"]:
        assert first["methods"][name]["selected_case_ids"] == second["methods"][name]["selected_case_ids"]
