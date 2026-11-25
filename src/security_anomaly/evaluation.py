"""Reproducible, incident-level comparisons for Zeek triage methods.

The evaluator keeps scoring and truth separate.  A label sidecar is read only
after all candidate scores have been produced, and every queue uses the same
case pool and review budget.  This makes the command suitable for evaluating
analyst workload without silently dropping cases that a method cannot score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__, periodic, zeek
from .paths import ensure_distinct
from .serialization import atomic_text_output
from .validation import InputContractError


METHODS = ("rare_periodic", "periodicity_only", "zeek_mad", "random")
LABEL_VALUES = frozenset(("target", "benign", "other"))


@dataclass(frozen=True)
class ReviewCase:
    """One independent queue item and its method-specific ranking scores."""

    case_id: str
    uids: tuple[str, ...]
    scores: Mapping[str, float | None]
    decisions: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.case_id, str):
            raise InputContractError("case_id must be a non-empty string")
        if not self.uids or any(not isinstance(uid, str) or not uid for uid in self.uids):
            raise InputContractError("a review case must contain non-empty observation UIDs")
        if len(set(self.uids)) != len(self.uids):
            raise InputContractError(f"case {self.case_id!r} contains a duplicate UID")
        for method, score in self.scores.items():
            if not isinstance(method, str) or not method:
                raise InputContractError("review score names must be non-empty strings")
            if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(float(score))):
                raise InputContractError(f"review score {method!r} must be finite or null")
        for method, decision in self.decisions.items():
            if not isinstance(method, str) or type(decision) is not bool:
                raise InputContractError("review decisions must map names to booleans")


def _label(value: Mapping[str, Any] | str) -> tuple[str, str | None]:
    if isinstance(value, str):
        label, episode = value, None
    elif isinstance(value, Mapping):
        label, episode = value.get("label"), value.get("episode")
    else:
        raise InputContractError("labels must be target, benign, or other objects")
    if not isinstance(label, str) or label not in LABEL_VALUES:
        raise InputContractError(f"label must be one of {sorted(LABEL_VALUES)}")
    if label == "target":
        if not isinstance(episode, str) or not episode.strip():
            raise InputContractError("target labels require a non-empty episode")
        return label, episode.strip()
    if episode is not None:
        raise InputContractError("only target labels may carry an episode")
    return label, None


def read_labels(path: str | Path, expected_uids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Read and validate one exact UID-to-truth JSONL sidecar.

    Every accepted observation UID must occur exactly once.  Extra UIDs are
    rejected, which prevents truth files from changing the denominator by
    accident.  The returned values are plain dictionaries so callers can
    serialize or inspect them without depending on a private class.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputContractError(f"labels input not found: {source}")
    expected = set(expected_uids)
    result: dict[str, dict[str, str]] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InputContractError(f"cannot read labels input {source}: {error}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if not expected:
            raise InputContractError("labels contain observations but the capture is empty")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise InputContractError(f"labels line {number} is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise InputContractError(f"labels line {number} must be an object")
        uid = payload.get("uid")
        if not isinstance(uid, str) or not uid:
            raise InputContractError(f"labels line {number} requires a non-empty uid")
        if uid not in expected:
            raise InputContractError(f"labels line {number} contains unknown uid {uid!r}")
        if uid in result:
            raise InputContractError(f"labels contain duplicate uid {uid!r}")
        label, episode = _label(payload)
        item = {"label": label}
        if episode is not None:
            item["episode"] = episode
        result[uid] = item
    missing = expected - result.keys()
    if missing:
        raise InputContractError(f"labels are missing {len(missing)} accepted observation UIDs")
    return result


def _case_truth(case: ReviewCase, labels: Mapping[str, Mapping[str, Any] | str]) -> dict[str, Any]:
    items = [_label(labels[uid]) if uid in labels else (None, None) for uid in case.uids]
    if any(label == "target" for label, _ in items):
        classification = "target"
    elif all(label == "benign" for label, _ in items):
        classification = "benign"
    elif any(label == "other" for label, _ in items):
        classification = "other"
    else:
        classification = "unknown"
    episodes = sorted({episode for label, episode in items if label == "target" and episode})
    return {"classification": classification, "episodes": episodes}


def _uniform_digest(seed: str, case_id: str) -> int:
    digest = hashlib.sha256(f"security-anomaly-evaluate-v1|{seed}|{case_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def _rank(cases: Sequence[ReviewCase], method: str, seed: str) -> list[ReviewCase]:
    if method == "random":
        return sorted(cases, key=lambda case: (-_uniform_digest(seed, case.case_id), case.case_id))
    return sorted(cases, key=lambda case: (-case.scores[method], case.case_id))


def rank_cases(
    cases: Sequence[ReviewCase],
    method: str,
    *,
    seed: str = "security-anomaly-ml",
) -> list[ReviewCase]:
    """Return a deterministic ranking for one method's available case pool.

    This is the shared ordering used by both labeled evaluation and the
    unlabeled analyst queue.  Missing scores must be filtered by the caller so
    the queue can report unavailable coverage instead of silently filling it.
    """

    if not isinstance(method, str) or not method:
        raise InputContractError("method must be a non-empty string")
    if not isinstance(seed, str) or not seed:
        raise InputContractError("seed must be a non-empty string")
    if any(case.scores.get(method) is None for case in cases):
        raise InputContractError(f"method {method!r} has unavailable scores in the ranking pool")
    return _rank(cases, method, seed)


def _episode_hit_probability(n: int, k: int, episode_cases: int) -> float:
    """Probability that a uniform sample of k distinct cases contains the episode."""
    if k > n - episode_cases:
        return 1.0
    # The chance of missing is C(n - episode_cases, k) / C(n, k).
    # Exchange sample and episode sizes to use the shorter product.
    draws, marked = sorted((k, episode_cases))
    return -math.expm1(sum(math.log1p(-marked / (n - i)) for i in range(draws)))


def _metrics(
    cases: Sequence[ReviewCase],
    selected: Sequence[ReviewCase],
    labels: Mapping[str, Mapping[str, Any] | str],
    *,
    method: str,
    budget: int,
    expected_selected: float | None = None,
) -> dict[str, Any]:
    all_truth = [_case_truth(case, labels) for case in cases]
    picked_truth = [_case_truth(case, labels) for case in selected]
    total_target_cases = sum(item["classification"] == "target" for item in all_truth)
    total_target_episodes = len(
        {
            episode
            for value in labels.values()
            for label, episode in (_label(value),)
            if label == "target" and episode
        }
    )
    target_cases = sum(item["classification"] == "target" for item in picked_truth)
    target_episodes = {episode for item in picked_truth for episode in item["episodes"]}
    pure_benign = sum(item["classification"] == "benign" for item in picked_truth)
    other = sum(item["classification"] == "other" for item in picked_truth)
    unknown = sum(item["classification"] == "unknown" for item in picked_truth)
    selected_count = len(selected)
    method_scores = [case.scores.get(method) for case in cases]
    selected_scores = [case.scores.get(method) for case in selected]
    score_available = [value for value in method_scores if value is not None]
    selected_unavailable = sum(value is None for value in selected_scores)
    selected_zero = sum(value is not None and float(value) == 0.0 for value in selected_scores)
    if method == "random":
        score_status = "randomized"
        available_count: int | None = None
        unavailable_count: int | None = None
    elif not score_available:
        score_status = "unavailable"
        available_count = 0
        unavailable_count = len(method_scores)
    elif len(score_available) < len(method_scores):
        score_status = "partial"
        available_count = len(score_available)
        unavailable_count = len(method_scores) - len(score_available)
    else:
        score_status = "complete"
        available_count = len(score_available)
        unavailable_count = 0
    return {
        "method": method,
        "budget": budget,
        "selected_count": selected_count,
        "selected_case_ids": [case.case_id for case in selected],
        "selected_target_cases": target_cases,
        "selected_target_episodes": len(target_episodes),
        "selected_episode_ids": sorted(target_episodes),
        "selected_pure_benign_cases": pure_benign,
        "selected_other_cases": other,
        "selected_unknown_cases": unknown,
        "score_status": score_status,
        "scored_case_count": available_count,
        "unavailable_case_count": unavailable_count,
        "selected_unavailable_fills": None if method == "random" else selected_unavailable,
        "selected_zero_score_fills": None if method == "random" else selected_zero,
        "total_target_cases": total_target_cases,
        "total_target_episodes": total_target_episodes,
        "total_benign_cases": sum(item["classification"] == "benign" for item in all_truth),
        "total_other_cases": sum(item["classification"] == "other" for item in all_truth),
        "total_unknown_cases": sum(item["classification"] == "unknown" for item in all_truth),
        "target_case_recall": target_cases / total_target_cases if total_target_cases else None,
        "target_episode_recall": len(target_episodes) / total_target_episodes if total_target_episodes else None,
        "selected_target_rate": target_cases / selected_count if selected_count else None,
        "selected_pure_benign_rate": pure_benign / selected_count if selected_count else None,
        "expected_target_cases_under_uniform": expected_selected,
    }


def compare_cases(
    cases: Sequence[ReviewCase],
    labels: Mapping[str, Mapping[str, Any] | str],
    budget: int,
    *,
    seed: str = "security-anomaly-ml",
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare all methods on one common pool with a deterministic budget."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise InputContractError("budget must be a non-negative integer")
    if not isinstance(seed, str) or not seed:
        raise InputContractError("seed must be a non-empty string")
    selected_methods = tuple(methods or METHODS)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise InputContractError("methods must contain at least one unique method")
    if any(not isinstance(method, str) or not method for method in selected_methods):
        raise InputContractError("methods must contain non-empty strings")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise InputContractError("case IDs must be unique")
    all_uids = [uid for case in cases for uid in case.uids]
    if len(set(all_uids)) != len(all_uids):
        raise InputContractError("one observation UID cannot belong to multiple review cases")
    unknown_labels = set(all_uids) - set(labels)
    if unknown_labels:
        raise InputContractError(f"labels are missing {len(unknown_labels)} case observation UIDs")
    numeric_methods = [
        method for method in selected_methods
        if method != "random" and any(c.scores.get(method) is not None for c in cases)
    ]
    pool = [c for c in cases if all(c.scores.get(method) is not None for method in numeric_methods)]
    effective_budget = min(budget, len(pool))
    pool_truth = [_case_truth(case, labels) for case in pool]
    eligible_targets = sum(truth["classification"] == "target" for truth in pool_truth)
    expected_target = eligible_targets * effective_budget / len(pool) if pool else 0.0
    episode_sizes = Counter(episode for truth in pool_truth for episode in truth["episodes"])
    expected_episodes = sum(
        frequency * _episode_hit_probability(len(pool), effective_budget, size)
        for size, frequency in Counter(episode_sizes.values()).items()
    )
    methods_payload: dict[str, Any] = {}
    reviewed: set[str] = set()
    for method in selected_methods:
        available = method == "random" or method in numeric_methods
        ranked = _rank(pool, method, seed) if available else []
        selected = ranked[:effective_budget]
        reviewed.update(case.case_id for case in selected)
        methods_payload[method] = _metrics(
            cases,
            selected,
            labels,
            method=method,
            budget=effective_budget,
            expected_selected=expected_target if available else None,
        )
        if not available:
            methods_payload[method]["target_case_recall"] = None
            methods_payload[method]["target_episode_recall"] = None
    return {
        "schema_version": "review-evaluation-v1",
        "case_count": len(cases),
        "common_pool_count": len(pool),
        "common_pool_target_cases": eligible_targets,
        "excluded_from_common_pool": len(cases) - len(pool),
        "uniform_expectation": {"target_cases": expected_target, "distinct_target_episodes": expected_episodes},
        "observation_count": len(all_uids),
        "budget_requested": budget,
        "budget": effective_budget,
        "seed": seed,
        "methods": methods_payload,
        "reviewed_cases": {
            c.case_id: {"uids": list(c.uids), "scores": dict(c.scores), **_case_truth(c, labels)}
            for c in sorted(pool, key=lambda case: case.case_id) if c.case_id in reviewed
        },
    }


def operating_point(
    cases: Sequence[ReviewCase],
    labels: Mapping[str, Mapping[str, Any] | str],
    *,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize threshold decisions over every case, independent of ranking."""

    selected_methods = tuple(
        methods
        or sorted({method for case in cases for method in case.decisions})
    )
    if len(set(selected_methods)) != len(selected_methods) or any(
        not isinstance(method, str) or not method for method in selected_methods
    ):
        raise InputContractError("methods must contain unique non-empty strings")
    result: dict[str, Any] = {}
    for method in selected_methods:
        selected = [case for case in cases if case.decisions.get(method, False)]
        result[method] = _metrics(
            cases,
            selected,
            labels,
            method=method,
            budget=len(cases),
            expected_selected=None,
        )
        if cases and result[method]["score_status"] == "unavailable":
            result[method]["target_case_recall"] = None
            result[method]["target_episode_recall"] = None
    return {"schema_version": "review-operating-point-v1", "case_count": len(cases), "methods": result}


def mad_witness(case: periodic.PeriodicCase, profile: zeek.ZeekProfile):
    """Return the highest scoring complete record, breaking member ties by UID."""
    scored = (
        (record, *zeek._score_value(record, profile))
        for record in case.records if record.complete
    )
    return max(scored, key=lambda item: (item[1], item[0].uid), default=None)


def score_zeek_case(case: periodic.PeriodicCase, profile: zeek.ZeekProfile | None) -> ReviewCase:
    witness = mad_witness(case, profile) if profile is not None else None
    mad_score = witness[1] if witness is not None else None
    return ReviewCase(
        case_id=case.case_id,
        uids=tuple(record.uid for record in case.records),
        scores={"rare_periodic": case.score, "periodicity_only": case.periodicity, "zeek_mad": mad_score},
        decisions={
            "rare_periodic": case.flagged,
            "periodicity_only": case.periodicity >= periodic.PERIODICITY_THRESHOLD,
            "zeek_mad": profile is not None and mad_score is not None and mad_score >= profile.threshold,
        },
    )


def _build_review_cases(reference: zeek.ZeekBatch, observed: zeek.ZeekBatch, local_nets: Sequence[str], sensor: str) -> tuple[list[ReviewCase], dict[str, Any]]:
    parsed, outbound_coverage = periodic.analyze_capture(reference, observed, list(local_nets), sensor)
    profile: zeek.ZeekProfile | None = None
    profile_unavailable_reason: str | None = None
    if parsed:
        try:
            profile = zeek.fit_profile(reference, sensor=sensor)
        except zeek.ZeekProfileError as error:
            profile_unavailable_reason = str(error)
    cases = [score_zeek_case(case, profile) for case in parsed]
    coverage = {
        "reference": reference.stats.__dict__,
        "observed": observed.stats.__dict__,
        **outbound_coverage,
        "cases": len(cases),
        "outbound_member_records": sum(len(case.uids) for case in cases),
        "mad_scored_cases": sum(case.scores["zeek_mad"] is not None for case in cases),
        "profile_fingerprint": profile.fingerprint if profile is not None else None,
        "profile_unavailable_reason": profile_unavailable_reason,
    }
    return cases, coverage


def _command_zeek(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_distinct(output, source, reference_path, labels_path)
    observed = zeek.read_zeek_jsonl(source)
    reference = zeek.read_zeek_jsonl(reference_path)
    cases, coverage = _build_review_cases(reference, observed, args.local_net, args.sensor)
    labels = read_labels(labels_path, [record.uid for record in observed.records])
    compared = compare_cases(cases, labels, args.budget, seed=args.seed)
    result = {
        "schema_version": "zeek-evaluation-v1",
        "product_version": __version__,
        "sensor": args.sensor,
        "parameters": {
            "budget": args.budget,
            "seed": args.seed,
            "local_nets": list(args.local_net),
            "session_gap_seconds": periodic.SESSION_GAP_SECONDS,
            "periodicity_threshold": periodic.PERIODICITY_THRESHOLD,
            "rarity_threshold": periodic.RARITY_THRESHOLD,
            "min_origin_history": periodic.MIN_HISTORY_RECORDS,
            "mad_threshold": zeek.DEFAULT_THRESHOLD,
        },
        "coverage": coverage,
        "operating_point": operating_point(cases, labels, methods=METHODS[:-1]),
        "comparison": compared,
        "labels": {"count": len(labels)},
    }
    with atomic_text_output(output) as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"command": "evaluate zeek", "cases": len(cases), "output": str(output)}, separators=(",", ":")), file=sys.stderr)
    return 0


def register_parser(subparsers: Any) -> None:
    """Register the evaluator and any optional dataset-specific evaluators."""

    evaluate = subparsers.add_parser("evaluate", help="compare incident-level triage methods")
    formats = evaluate.add_subparsers(dest="evaluation_format", required=True)
    zeek_parser = formats.add_parser("zeek", help="compare Zeek incident ranking methods")
    zeek_parser.add_argument("input", type=Path, help="observed Zeek JSONL")
    zeek_parser.add_argument("--reference", required=True, type=Path, help="earlier reference Zeek JSONL")
    zeek_parser.add_argument("--labels", required=True, type=Path, help="exact UID truth sidecar JSONL")
    zeek_parser.add_argument("--sensor", required=True, help="stable sensor identifier")
    zeek_parser.add_argument("--local-net", action="append", required=True, help="local CIDR, repeatable")
    zeek_parser.add_argument("--budget", type=int, default=10, help="maximum cases in each review queue")
    zeek_parser.add_argument("--seed", default="security-anomaly-ml", help="stable queue seed")
    zeek_parser.add_argument("--output", required=True, type=Path, help="evaluation JSON output")
    zeek_parser.set_defaults(handler=_command_zeek)
    from . import cic_evaluation

    cic_evaluation.register_parser(formats)


__all__ = [
    "METHODS",
    "ReviewCase",
    "compare_cases",
    "operating_point",
    "rank_cases",
    "read_labels",
    "register_parser",
]
