"""Explainable periodic outbound-connection triage for Zeek JSONL.

The detector is deliberately a small policy-driven rule.  It learns no model
and does not use labels.  A reference capture supplies the per-origin
destination frequency used for rarity.  An observed capture is partitioned
into deterministic sessions by origin, destination, protocol and destination
port.  A session is flagged only when its starts are regular and its
destination is uncommon for that origin.

The caller supplies the sensor's local CIDRs. Link-local destinations stay
on the originating link and are excluded from outbound sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from . import __version__
from .ocsf import detection_finding
from .paths import ensure_distinct
from .serialization import write_incidents_jsonl
from .validation import InputContractError
from .zeek import (
    ZeekBatch,
    ZeekInputError,
    ZeekRecord,
    _median as _zeek_median,
    _validate_sensor,
    read_zeek_jsonl,
)


SCHEMA_VERSION = "zeek-periodic-v1"
DETECTOR_NAME = "zeek-periodic-outbound"
DETECTOR_VERSION = "1"
SESSION_GAP_SECONDS = 1800.0
PERIODICITY_THRESHOLD = 0.8
RARITY_THRESHOLD = 0.8
MIN_HISTORY_RECORDS = 3
_LIMITED_BROADCAST = ipaddress.IPv4Address("255.255.255.255")
Network = ipaddress.IPv4Network | ipaddress.IPv6Network


class PeriodicInputError(InputContractError):
    """Raised when the periodic detector inputs or policy are invalid."""


@dataclass(frozen=True)
class PeriodicCase:
    """One deterministic observed outbound session and its rule evidence."""

    case_id: str
    records: tuple[ZeekRecord, ...]
    source_ip: str
    destination_ip: str
    protocol: str
    destination_port: int
    periodicity: float
    rarity: float
    score: float
    flagged: bool
    reference_origin_count: int
    reference_destination_count: int
    distinct_start_count: int
    interval_median_seconds: float
    interval_mean_absolute_error_seconds: float


def _utc_timestamp(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PeriodicInputError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _network_list(values: Sequence[str]) -> tuple[Network, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise PeriodicInputError("at least one local CIDR is required")
    networks: set[Network] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise PeriodicInputError("local CIDRs must be non-empty strings")
        try:
            network = ipaddress.ip_network(value.strip(), strict=False)
        except ValueError as error:
            raise PeriodicInputError(f"invalid local CIDR {value!r}") from error
        networks.add(network)
    return tuple(sorted(networks, key=lambda net: (net.version, int(net.network_address), net.prefixlen)))


def _ip(value: str, field: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(value)
    except ValueError as error:
        raise PeriodicInputError(f"{field} is not a valid IP address") from error


def _eligible_outbound(record: ZeekRecord, networks: Sequence[Network]) -> bool:
    source = _ip(record.src_ip, "source address")
    destination = _ip(record.dst_ip, "destination address")
    if not any(source in network for network in networks):
        return False
    if any(destination in network for network in networks):
        return False
    if destination.is_multicast or destination.is_unspecified or destination.is_loopback or destination.is_link_local:
        return False
    if destination == _LIMITED_BROADCAST:
        return False
    return True


def _validate_batches(reference: ZeekBatch, observed: ZeekBatch) -> None:
    if not isinstance(reference, ZeekBatch) or not isinstance(observed, ZeekBatch):
        raise PeriodicInputError("reference and observed inputs must be ZeekBatch values")
    reference_uids = [record.uid for record in reference.records]
    observed_uids = [record.uid for record in observed.records]
    if len(reference_uids) != len(set(reference_uids)):
        raise PeriodicInputError("reference contains duplicate Zeek UIDs")
    if len(observed_uids) != len(set(observed_uids)):
        raise PeriodicInputError("observed input contains duplicate Zeek UIDs")
    overlap = set(reference_uids).intersection(observed_uids)
    if overlap:
        raise PeriodicInputError("reference and observed inputs must have disjoint Zeek UIDs")
    if not reference.records or not observed.records:
        return
    reference_times = [_utc_timestamp(record.timestamp, "reference timestamp") for record in reference.records]
    observed_times = [_utc_timestamp(record.timestamp, "observed timestamp") for record in observed.records]
    if max(reference_times) >= min(observed_times):
        raise PeriodicInputError(
            "reference timestamps must be strictly earlier than observed timestamps"
        )


def _start_intervals(records: Sequence[ZeekRecord]) -> list[float]:
    starts = sorted({_utc_timestamp(record.timestamp, "observed timestamp") for record in records})
    return [(later - earlier).total_seconds() for earlier, later in pairwise(starts)]


def _periodicity(records: Sequence[ZeekRecord]) -> tuple[float, int, float, float]:
    intervals = _start_intervals(records)
    distinct_count = len(intervals) + 1
    if len(intervals) < 3:
        return 0.0, distinct_count, 0.0, 0.0
    median = _zeek_median(intervals)
    error = sum(abs(interval - median) for interval in intervals) / len(intervals)
    score = 1.0 / (1.0 + error / median)
    return score, distinct_count, median, error


def _case_id(sensor: str, first: ZeekRecord) -> str:
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "sensor": sensor,
        "source_ip": first.src_ip,
        "destination_ip": first.dst_ip,
        "protocol": first.protocol,
        "destination_port": first.dst_port,
        "first_start": _utc_timestamp(first.timestamp, "observed timestamp").isoformat(
            timespec="microseconds"
        ),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "pc_" + hashlib.sha256(encoded).hexdigest()


def _session_groups(
    records: Iterable[ZeekRecord],
) -> list[tuple[ZeekRecord, ...]]:
    by_key: dict[tuple[str, str, str, int], list[ZeekRecord]] = defaultdict(list)
    for record in records:
        by_key[(record.src_ip, record.dst_ip, record.protocol, record.dst_port)].append(record)
    groups: list[tuple[ZeekRecord, ...]] = []
    for key in sorted(by_key):
        ordered = sorted(
            by_key[key],
            key=lambda record: (_utc_timestamp(record.timestamp, "observed timestamp"), record.uid),
        )
        current: list[ZeekRecord] = []
        previous: datetime | None = None
        for record in ordered:
            timestamp = _utc_timestamp(record.timestamp, "observed timestamp")
            if previous is not None and (timestamp - previous).total_seconds() > SESSION_GAP_SECONDS:
                groups.append(tuple(current))
                current = []
            current.append(record)
            previous = timestamp
        if current:
            groups.append(tuple(current))
    groups.sort(
        key=lambda group: (
            group[0].src_ip,
            group[0].dst_ip,
            group[0].protocol,
            group[0].dst_port,
            _utc_timestamp(group[0].timestamp, "observed timestamp"),
            tuple(record.uid for record in group),
        )
    )
    return groups


def build_cases(
    reference: ZeekBatch,
    observed: ZeekBatch,
    local_nets: list[str],
    sensor: str,
) -> list[PeriodicCase]:
    """Build all eligible observed sessions with deterministic rule evidence."""

    cases, _ = analyze_capture(reference, observed, local_nets, sensor)
    return cases


def analyze_capture(
    reference: ZeekBatch, observed: ZeekBatch, local_nets: list[str], sensor: str,
) -> tuple[list[PeriodicCase], dict]:
    try:
        sensor = _validate_sensor(sensor)
    except ZeekInputError as error:
        raise PeriodicInputError(str(error)) from error
    networks = _network_list(local_nets)
    _validate_batches(reference, observed)
    history = [record for record in reference.records if _eligible_outbound(record, networks)]
    outbound = [record for record in observed.records if _eligible_outbound(record, networks)]
    origin_total = Counter(record.src_ip for record in history)
    origin_destination = Counter((record.src_ip, record.dst_ip) for record in history)
    cases: list[PeriodicCase] = []
    for records in _session_groups(outbound):
        first = records[0]
        periodicity, distinct_count, median_interval, mean_absolute_error = _periodicity(records)
        total = origin_total[first.src_ip]
        destination_count = origin_destination[(first.src_ip, first.dst_ip)]
        rarity = (
            1.0 - destination_count / total
            if total >= MIN_HISTORY_RECORDS
            else 0.0
        )
        score = periodicity * rarity
        cases.append(
            PeriodicCase(
                case_id=_case_id(sensor, first),
                records=records,
                source_ip=first.src_ip,
                destination_ip=first.dst_ip,
                protocol=first.protocol,
                destination_port=first.dst_port,
                periodicity=periodicity,
                rarity=rarity,
                score=score,
                flagged=periodicity >= PERIODICITY_THRESHOLD and rarity >= RARITY_THRESHOLD,
                reference_origin_count=total,
                reference_destination_count=destination_count,
                distinct_start_count=distinct_count,
                interval_median_seconds=median_interval,
                interval_mean_absolute_error_seconds=mean_absolute_error,
            )
        )
    coverage = {
        "eligible_reference_outbound_records": len(history),
        "eligible_observed_outbound_records": len(outbound),
        "reference_origins": len(origin_total),
        "reference_origins_with_min_history": sum(n >= MIN_HISTORY_RECORDS for n in origin_total.values()),
        "empty_reference_window": not reference.records,
        "empty_eligible_reference_history": not history,
    }
    return cases, coverage


def case_evidence(case: PeriodicCase) -> dict[str, Any]:
    records = case.records
    first_seen = records[0].timestamp.astimezone(UTC)
    last_seen = max(record.end_time.astimezone(UTC) for record in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "detector": DETECTOR_NAME,
        "algorithm": "periodicity * destination-rarity",
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "periodicity_threshold": PERIODICITY_THRESHOLD,
        "rarity_threshold": RARITY_THRESHOLD,
        "min_history_records": MIN_HISTORY_RECORDS,
        "periodicity": round(case.periodicity, 12),
        "rarity": round(case.rarity, 12),
        "score": round(case.score, 12),
        "distinct_start_count": case.distinct_start_count,
        "first_seen": first_seen.isoformat(timespec="microseconds"),
        "last_seen": last_seen.isoformat(timespec="microseconds"),
        "intervals_seconds": _start_intervals(records),
        "interval_median_seconds": round(case.interval_median_seconds, 12),
        "interval_mean_absolute_error_seconds": round(case.interval_mean_absolute_error_seconds, 12),
        "reference_origin_count": case.reference_origin_count,
        "reference_destination_count": case.reference_destination_count,
        "observed_count": len(records),
        "protocol": case.protocol,
        "observed_uids": [record.uid for record in records],
        "source_ports": sorted({record.src_port for record in records}),
    }


def _finding(case: PeriodicCase, sensor: str) -> dict[str, Any]:
    first_seen = case.records[0].timestamp.astimezone(UTC)
    last_seen = max(record.end_time for record in case.records).astimezone(UTC)
    return detection_finding(
        sensor=sensor,
        observation_id=case.case_id,
        first_seen=first_seen,
        last_seen=last_seen,
        src_ip=case.source_ip,
        dst_ip=case.destination_ip,
        dst_port=case.destination_port,
        detector=DETECTOR_NAME,
        detector_version=DETECTOR_VERSION,
        title="Periodic outbound connection to an uncommon destination",
        evidence=case_evidence(case),
        product_version=__version__,
        group_key=[SCHEMA_VERSION, sensor, case.case_id],
    )


def findings_for_cases(cases: Iterable[PeriodicCase], sensor: str) -> tuple[dict[str, Any], ...]:
    """Return only flagged OCSF findings while preserving deterministic order."""

    sensor = _validate_sensor(sensor)
    return tuple(_finding(case, sensor) for case in cases if case.flagged)


def _command_periodic(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    ensure_distinct(output, args.input, args.reference)
    reference = read_zeek_jsonl(args.reference)
    observed = read_zeek_jsonl(args.input)
    cases, coverage = analyze_capture(reference, observed, args.local_net, args.sensor)
    findings = findings_for_cases(cases, args.sensor)
    write_incidents_jsonl(output, findings)
    summary = {
        "command": "zeek-periodic",
        "sensor": args.sensor,
        "reference": asdict(reference.stats),
        "observed": asdict(observed.stats),
        "coverage": coverage,
        "eligible_sessions": len(cases),
        "flagged_sessions": len(findings),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, separators=(",", ":")), file=sys.stderr)
    return 0


def register_parser(subparsers: Any) -> None:
    """Register the ``zeek-periodic`` command with the product CLI."""

    command = subparsers.add_parser(
        "zeek-periodic",
        help="find periodic outbound sessions to uncommon destinations",
    )
    command.add_argument("input", type=Path, help="observed Zeek JSONL input")
    command.add_argument(
        "--reference", required=True, type=Path, help="earlier reference Zeek JSONL input"
    )
    command.add_argument(
        "--local-net",
        action="append",
        required=True,
        metavar="CIDR",
        help="local CIDR, repeat for each sensor-local network",
    )
    command.add_argument("--sensor", required=True, help="stable sensor identifier")
    command.add_argument("--output", required=True, type=Path, help="OCSF findings JSONL output")
    command.set_defaults(handler=_command_periodic)


__all__ = [
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "MIN_HISTORY_RECORDS",
    "PERIODICITY_THRESHOLD",
    "PeriodicCase",
    "PeriodicInputError",
    "RARITY_THRESHOLD",
    "SCHEMA_VERSION",
    "SESSION_GAP_SECONDS",
    "analyze_capture",
    "build_cases",
    "findings_for_cases",
    "register_parser",
]
