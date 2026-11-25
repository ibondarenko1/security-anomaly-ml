"""Deterministic analyst queues for unlabeled Zeek connection captures.

The review queue scores every eligible observed session, keeps only cases for
which the selected method has a score, and returns the highest scoring cases
for a bounded analyst budget.  It deliberately has no label input.  A later
verdict sidecar can therefore be bound to the exact snapshot emitted here
without changing the ranking or pretending that a queue is an evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

from . import __version__, evaluation, periodic, zeek
from .paths import ensure_distinct
from .serialization import atomic_text_output
from .validation import InputContractError


SCHEMA_VERSION = "zeek-review-v1"
METHODS = ("zeek_mad", "rare_periodic", "periodicity_only")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_fingerprint(row: Mapping[str, Any]) -> str:
    """Return the stable identity of one delivered case snapshot.

    The fingerprint includes ordered membership and the complete diagnostic
    evidence.  It is stable when only queue rank or budget changes and changes
    when a logical session is extended or any score input is changed.
    """

    if not isinstance(row, Mapping):
        raise InputContractError("review snapshot must be a JSON object")
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"snapshot_id", "rank", "product_version"}
    }
    return "snap_" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _timestamp(value: Any) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _mad_evidence(case: periodic.PeriodicCase, profile: zeek.ZeekProfile) -> dict[str, Any]:
    # Ranking has already established that at least one member was scorable.
    record, score, deviations = evaluation.mad_witness(case, profile)
    return {
        "algorithm": "max member robust-median-mad score",
        "profile_fingerprint": profile.fingerprint,
        "profile_threshold": profile.threshold,
        "fit_records": profile.fit_records,
        "scored_member_count": sum(item.complete for item in case.records),
        "maximum_member": {
            "uid": record.uid, "timestamp": _timestamp(record.timestamp), "score": score,
            "measured": dict(record.features), "deviations": deviations,
            "strongest_features": list(zeek.explain_score(record, profile, score=score, deviations=deviations)),
            "anomalous_features": [name for name in profile.feature_names if deviations[name] >= profile.threshold],
        },
        "observed_uids": [item.uid for item in case.records],
    }


def _member_digest(records: Sequence[zeek.ZeekRecord]) -> str:
    """Bind revisions to every normalized observation, including incomplete members."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json({
            "uid": record.uid, "ts": _timestamp(record.timestamp),
            "src": [record.src_ip, record.src_port], "dst": [record.dst_ip, record.dst_port],
            "proto": record.protocol, "features": dict(record.features),
        }))
        digest.update(b"\n")
    return digest.hexdigest()


def _method_record(
    case: periodic.PeriodicCase,
    *,
    sensor: str,
    method: str,
    score: float,
    threshold: Any,
    threshold_decision: bool,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    records = case.records
    first_seen = min(record.timestamp for record in records).astimezone(UTC)
    last_seen = max(record.end_time for record in records).astimezone(UTC)
    evidence_payload = dict(evidence)
    uids = [record.uid for record in records]
    endpoints = {
        "source_ip": case.source_ip,
        "destination_ip": case.destination_ip,
        "protocol": case.protocol,
        "destination_port": case.destination_port,
        "source_ports": sorted({record.src_port for record in records}),
    }
    row = {
        "schema_version": SCHEMA_VERSION,
        "product_version": __version__,
        "rank": 0,
        "case_id": case.case_id,
        "sensor": sensor,
        "method": method,
        "score": score,
        "threshold": threshold,
        "threshold_decision": threshold_decision,
        "time": {"first_seen": _timestamp(first_seen), "last_seen": _timestamp(last_seen)},
        "endpoints": endpoints,
        "uids": uids,
        "members_sha256": _member_digest(records),
        "diagnostic": {
            "reason": (
                "case score meets the detector threshold"
                if threshold_decision
                else "case ranked for review below the detector threshold"
            ),
            "evidence": evidence_payload,
        },
    }
    row["snapshot_id"] = snapshot_fingerprint(row)
    return row


def build_queue(
    reference: zeek.ZeekBatch,
    observed: zeek.ZeekBatch,
    local_nets: Sequence[str],
    sensor: str,
    *,
    method: str = "zeek_mad",
    budget: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic top-K queue and method-specific coverage."""

    if method not in METHODS:
        raise InputContractError(f"method must be one of {', '.join(METHODS)}")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise InputContractError("budget must be a non-negative integer")
    try:
        sensor = zeek._validate_sensor(sensor)
    except zeek.ZeekInputError as error:
        raise InputContractError(str(error)) from error
    cases, capture_coverage = periodic.analyze_capture(reference, observed, list(local_nets), sensor)
    profile: zeek.ZeekProfile | None = None
    profile_error: str | None = None
    if method == "zeek_mad":
        try:
            profile = zeek.fit_profile(reference, sensor=sensor)
        except zeek.ZeekProfileError as error:
            profile_error = str(error)

    case_by_id = {case.case_id: case for case in cases}
    scored = [evaluation.score_zeek_case(case, profile) for case in cases]
    available = [case for case in scored if case.scores[method] is not None]
    ranked = evaluation.rank_cases(available, method)
    selected = []
    for rank, item in enumerate(ranked[:budget], 1):
        case = case_by_id[item.case_id]
        if method == "zeek_mad":
            assert profile is not None  # An available MAD score requires a fitted profile.
            threshold = profile.threshold
            evidence = _mad_evidence(case, profile)
        else:
            evidence = periodic.case_evidence(case)
            if method == "periodicity_only":
                threshold = periodic.PERIODICITY_THRESHOLD
                evidence.update(algorithm="start-interval regularity", score=case.periodicity)
            else:
                threshold = {"periodicity": periodic.PERIODICITY_THRESHOLD, "rarity": periodic.RARITY_THRESHOLD}
        row = _method_record(case, sensor=sensor, method=method, score=item.scores[method],
                             threshold=threshold, threshold_decision=item.decisions[method], evidence=evidence)
        row["rank"] = rank
        selected.append(row)
    coverage = {
        "eligible_case_count": len(cases),
        "available_case_count": len(available),
        "unavailable_case_count": len(cases) - len(available),
        "zero_score_case_count": sum(item.scores[method] == 0.0 for item in available),
        "selected_case_count": len(selected),
        "budget_requested": budget,
        "budget": len(selected),
        "profile_fingerprint": profile.fingerprint if profile is not None else None,
        "profile_unavailable_reason": profile_error,
        "capture": capture_coverage,
    }
    return selected, coverage


def _command_zeek(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ensure_distinct(output, source, reference)
    observed_batch = zeek.read_zeek_jsonl(source)
    reference_batch = zeek.read_zeek_jsonl(reference)
    queue, coverage = build_queue(
        reference_batch,
        observed_batch,
        args.local_net,
        args.sensor,
        method=args.method,
        budget=args.budget,
    )
    with atomic_text_output(output) as handle:
        for item in queue:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            handle.write("\n")
    print(
        json.dumps(
            {
                "command": "review zeek",
                "sensor": args.sensor,
                "method": args.method,
                "coverage": coverage,
                "output": str(output),
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    return 0


def register_parser(subparsers: Any) -> None:
    """Register unlabeled review queues and the optional verdict subcommand."""

    review = subparsers.add_parser("review", help="build bounded analyst review queues")
    formats = review.add_subparsers(dest="review_format", required=True)
    zeek_parser = formats.add_parser("zeek", help="rank unlabeled Zeek sessions")
    zeek_parser.add_argument("input", type=Path, help="observed Zeek JSONL")
    zeek_parser.add_argument("--reference", required=True, type=Path, help="earlier reference Zeek JSONL")
    zeek_parser.add_argument("--sensor", required=True, help="stable sensor identifier")
    zeek_parser.add_argument("--local-net", action="append", required=True, metavar="CIDR")
    zeek_parser.add_argument("--method", choices=METHODS, default="zeek_mad")
    zeek_parser.add_argument("--budget", type=int, default=10)
    zeek_parser.add_argument("--output", required=True, type=Path)
    zeek_parser.set_defaults(handler=_command_zeek)
    from .review_verdict import register_verdict_parser

    register_verdict_parser(formats)


__all__ = ["METHODS", "SCHEMA_VERSION", "build_queue", "register_parser", "snapshot_fingerprint"]
