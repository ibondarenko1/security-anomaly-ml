"""Local OCSF 1.6 Detection Findings from sensor observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from functools import cache
from ipaddress import ip_address
from pathlib import Path
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator, ValidationError

from . import __version__
from .package_resources import INCIDENT_SCHEMA_RESOURCE, resource_json
from .paths import ensure_distinct
from .serialization import write_incidents_jsonl
from .timeutils import source_utc, source_zone
from .validation import InputContractError

OCSF_VERSION = "1.6.0"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _milliseconds(value: datetime) -> int:
    if value.utcoffset() is None:
        raise InputContractError("OCSF event times require a source timezone")
    delta = value.astimezone(UTC) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def detection_finding(
    *,
    sensor: str,
    observation_id: str,
    first_seen: datetime,
    last_seen: datetime,
    src_ip: str,
    dst_ip: str,
    dst_port: int,
    detector: str,
    detector_version: str,
    title: str,
    evidence: dict,
    product_version: str,
    src_port: int | None = None,
    group_key: list | None = None,
) -> dict:
    """Represent one finding snapshot. Its logical ID survives a later end time."""
    if not sensor.strip() or sensor.strip() != sensor or not observation_id:
        raise InputContractError("Supply a non-empty sensor and observation identifier")
    if last_seen < first_seen:
        raise InputContractError("Finding ends before its first observation")
    try:
        source = {"ip": str(ip_address(src_ip))}
        destination = {"ip": str(ip_address(dst_ip)), "port": dst_port}
    except ValueError as error:
        raise InputContractError(f"Invalid finding endpoint: {error}") from error
    if src_port is not None:
        source["port"] = src_port
    for port in (dst_port, src_port):
        if port is not None and (type(port) is not int or not 0 <= port <= 65535):
            raise InputContractError("Finding port must be an integer in [0, 65535]")
    start, end = _milliseconds(first_seen), _milliseconds(last_seen)
    identity = _digest(
        [sensor, detector, group_key if group_key is not None else observation_id]
    )
    result = {
        "category_uid": 2,
        "class_uid": 2004,
        "activity_id": 1,
        "type_uid": 200401,
        "severity_id": 1,
        "time": end,
        "start_time": start,
        "end_time": end,
        "metadata": {
            "version": OCSF_VERSION,
            "product": {"name": "Security Anomaly ML", "version": product_version},
            "log_provider": sensor,
        },
        "finding_info": {
            "uid": identity,
            "title": title,
            "analytic": {
                "name": detector,
                "version": detector_version,
                "type_id": 4 if detector == "context-rf-v2" else 3,
            },
        },
        "evidences": [
            {
                "uid": observation_id,
                "src_endpoint": source,
                "dst_endpoint": destination,
                "data": evidence,
            }
        ],
    }
    result["metadata"]["uid"] = _digest(result)
    return result


@cache
def _incident_validator() -> Draft202012Validator:
    return Draft202012Validator(resource_json(INCIDENT_SCHEMA_RESOURCE))


def incident_finding(
    record: dict,
    sensor: str,
    zone: ZoneInfo,
    *,
    flow_ids: list[str] | None = None,
) -> dict:
    """Export a promoted incident with its decision evidence and original identifier."""
    try:
        _incident_validator().validate(record)
        start = source_utc(datetime.fromisoformat(record["first_seen"]), zone)
        end = source_utc(datetime.fromisoformat(record["last_seen"]), zone)
    except (ValidationError, ValueError, OverflowError) as error:
        message = error.message if isinstance(error, ValidationError) else str(error)
        raise InputContractError(f"Invalid incident-v1: {message}") from error
    maximum = record["max_attack_score"]
    mean = record["mean_attack_score"]
    if not (
        math.isfinite(maximum)
        and math.isfinite(mean)
        and mean <= maximum
        and maximum >= 0.25
    ):
        raise InputContractError(
            "Scores do not describe a promoted frozen-model incident"
        )
    evidence = {
        "feature_contract": record["feature_contract"],
        "aggregation_policy": "B",
        "inactivity_gap_seconds": 300,
        "flow_threshold": 0.10,
        "promotion_threshold": 0.25,
        "flow_count": record["flow_count"],
        "protocols": record["protocols"],
        "max_attack_score": maximum,
        "mean_attack_score": mean,
    }
    if flow_ids:
        evidence["flow_ids"] = sorted(set(flow_ids))
    # The legacy incident_id includes last_seen. The SOC finding groups snapshots
    # by the Policy B endpoints and the first instant in UTC across rotations.
    try:
        key = [
            str(ip_address(record["src_ip"])),
            str(ip_address(record["dst_ip"])),
            record["dst_port"],
            _milliseconds(start),
        ]
    except ValueError as error:
        raise InputContractError(f"Invalid incident address: {error}") from error
    return detection_finding(
        sensor=sensor,
        observation_id=record["incident_id"],
        first_seen=start,
        last_seen=end,
        src_ip=key[0],
        dst_ip=key[1],
        dst_port=key[2],
        detector=record["model_version"],
        detector_version=record["model_version"],
        title="Promoted network-flow incident",
        evidence=evidence,
        product_version=__version__,
        group_key=key,
    )


def _command_export(args: argparse.Namespace) -> int:
    ensure_distinct(args.output, args.input)
    zone = source_zone(args.timezone)
    if not args.sensor.strip() or args.sensor.strip() != args.sensor:
        raise InputContractError(
            "Supply a non-empty sensor identifier without surrounding spaces"
        )

    def findings():
        try:
            with args.input.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        yield incident_finding(json.loads(line), args.sensor, zone)
                    except (ValueError, OverflowError) as error:
                        raise InputContractError(
                            f"Incident line {number}: {error}"
                        ) from error
        except (OSError, UnicodeError) as error:
            raise InputContractError(f"Cannot read incident input: {error}") from error

    write_incidents_jsonl(args.output, findings())
    return 0


def register_parser(subparsers) -> None:
    command = subparsers.add_parser(
        "export-ocsf", help="map incident-v1 JSONL to OCSF Detection Findings"
    )
    command.add_argument("input", type=Path)
    command.add_argument("--sensor", required=True, help="stable sensor identifier")
    command.add_argument(
        "--timezone",
        required=True,
        help="IANA timezone of input timestamps; UTC for replay output",
    )
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(handler=_command_export)
