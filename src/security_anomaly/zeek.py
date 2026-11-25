"""Local, interpretable triage for Zeek JSON ``conn.log`` records.

This module deliberately does not reuse the CICFlowMeter detector.  Zeek's
JSON fields are parsed into a small sensor-local profile made from robust
medians and median absolute deviations (MAD).  A profile is fitted from one
input and a separate input is scored against it; no labels, network access,
or feature imputation are involved.

"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .ocsf import detection_finding
from .paths import ensure_distinct
from .serialization import atomic_text_output, write_incidents_jsonl
from .validation import InputContractError

SCHEMA_VERSION = "zeek-profile-v1"
DETECTOR_NAME = "zeek-profile"
DEFAULT_THRESHOLD = 6.0
MIN_DEFAULT_RECORDS = 3
MAD_SCALE_FACTOR = 1.4826
MAD_SCALE_FLOOR = 1e-9

# These are deliberately Zeek-native fields.  In particular, this list is
# not a projection of the CICFlowMeter feature contract.
PROFILE_FEATURES: tuple[str, ...] = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
)
_INTEGER_FEATURES = frozenset(
    {
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "missed_bytes",
    }
)

_REQUIRED_FIELDS = (
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
)
_SENSOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


class ZeekError(InputContractError):
    """Base class for fail-closed Zeek input/profile errors."""


class ZeekInputError(ZeekError):
    """Raised when a Zeek JSONL source cannot be read or is not usable."""


class ZeekIncompleteRecord(ZeekInputError):
    """Raised for a record missing required identity fields."""


class ZeekProfileError(ZeekError):
    """Raised when a fitted profile is missing or incompatible."""


@dataclass(frozen=True)
class ZeekRecord:
    """One metadata-valid Zeek connection, possibly incomplete for scoring."""

    uid: str
    timestamp: datetime
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    features: Mapping[str, float]
    missing_features: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether all profile features were present and finite."""

        return not self.missing_features

    @property
    def end_time(self) -> datetime:
        """Connection end in UTC when Zeek supplied a duration."""

        duration = self.features.get("duration")
        if duration is None:
            return self.timestamp
        try:
            return self.timestamp + timedelta(seconds=duration)
        except (OverflowError, ValueError) as error:
            raise ZeekInputError("duration places the connection outside the UTC range") from error


@dataclass(frozen=True)
class ZeekParseStats:
    """Counters that make exclusions visible to a caller and to a SOC runbook."""

    lines_read: int
    parsed_records: int
    scorable_records: int
    excluded_records: int
    incomplete_records: int
    invalid_records: int
    duplicate_records: int

    @property
    def not_scorable_records(self) -> int:
        return self.lines_read - self.scorable_records


@dataclass(frozen=True)
class ZeekBatch:
    """Parsed records and their validation counters."""

    records: tuple[ZeekRecord, ...]
    stats: ZeekParseStats

    @property
    def complete_records(self) -> tuple[ZeekRecord, ...]:
        return tuple(record for record in self.records if record.complete)


@dataclass(frozen=True)
class ZeekProfile:
    """Sensor-scoped robust baseline used by ``zeek-score``."""

    schema_version: str
    product_version: str
    sensor: str
    feature_names: tuple[str, ...]
    center: Mapping[str, float]
    scale: Mapping[str, float]
    fit_records: int
    threshold: float
    # Raw MAD values are retained for operator-facing explanations.  ``None``
    # denotes a profile written before this optional field was introduced.
    # The fitted score and profile fingerprint continue to use ``scale`` so
    # old profile files remain valid and produce identical scores.
    mad: Mapping[str, float] | None = None

    @property
    def fingerprint(self) -> str:
        """Stable digest of the sensor-scoped fitted profile."""

        canonical = {
            "schema_version": self.schema_version,
            "sensor": self.sensor,
            "feature_names": list(self.feature_names),
            "center": {name: self.center[name] for name in self.feature_names},
            "scale": {name: self.scale[name] for name in self.feature_names},
            "fit_records": self.fit_records,
            "threshold": self.threshold,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def mad_fingerprint(self) -> str | None:
        if self.mad is None:
            return None
        encoded = json.dumps([self.mad[name] for name in self.feature_names], allow_nan=False,
                             separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "product_version": self.product_version,
            "detector": DETECTOR_NAME,
            "sensor": self.sensor,
            "feature_names": list(self.feature_names),
            "center": {name: self.center[name] for name in self.feature_names},
            "scale": {name: self.scale[name] for name in self.feature_names},
            "fit_records": self.fit_records,
            "threshold": self.threshold,
            "profile_fingerprint": self.fingerprint,
            "algorithm": {
                "name": "robust-median-mad",
                "score": "max(abs(value - median) / max(1.4826 * MAD, 1e-9))",
                "imputation": "none",
            },
        }
        if self.mad is not None:
            payload["mad"] = {
                name: self.mad[name] for name in self.feature_names
            }
            payload["mad_sha256"] = self.mad_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ZeekProfile:
        if not isinstance(payload, Mapping):
            raise ZeekProfileError("profile must be a JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ZeekProfileError(
                f"unsupported Zeek profile schema: {payload.get('schema_version')!r}"
            )
        sensor = _validate_sensor(payload.get("sensor"))
        feature_names_value = payload.get("feature_names")
        if feature_names_value != list(PROFILE_FEATURES):
            raise ZeekProfileError("profile feature_names do not match Zeek profile schema")
        try:
            center = _finite_mapping(payload.get("center"), "center", positive=False)
            scale = _finite_mapping(payload.get("scale"), "scale", positive=True)
            mad_value = payload.get("mad")
            mad = (
                None
                if mad_value is None
                else _finite_mapping(mad_value, "mad", positive=False)
            )
        except ZeekInputError as error:
            raise ZeekProfileError(str(error)) from error
        if tuple(center) != PROFILE_FEATURES or tuple(scale) != PROFILE_FEATURES:
            raise ZeekProfileError("profile feature mappings are incomplete or reordered")
        if mad is not None and tuple(mad) != PROFILE_FEATURES:
            raise ZeekProfileError("profile mad mapping is incomplete or reordered")
        if mad is not None and any(value < 0 for value in mad.values()):
            raise ZeekProfileError("profile mad values must be non-negative")
        if mad is not None and any(
            scale[name] != max(MAD_SCALE_FACTOR * mad[name], MAD_SCALE_FLOOR) for name in PROFILE_FEATURES
        ):
            raise ZeekProfileError("profile MAD is inconsistent with the stored robust scale")
        fit_records = payload.get("fit_records")
        if isinstance(fit_records, bool) or not isinstance(fit_records, int) or fit_records < 1:
            raise ZeekProfileError("fit_records must be a positive integer")
        try:
            threshold = _finite_number(payload.get("threshold"), "threshold")
        except ZeekInputError as error:
            raise ZeekProfileError(str(error)) from error
        if threshold < 0:
            raise ZeekProfileError("threshold must be non-negative")
        product_version = payload.get("product_version")
        if not isinstance(product_version, str) or not product_version.strip():
            raise ZeekProfileError("profile product_version must be non-empty")
        if payload.get("detector") != DETECTOR_NAME:
            raise ZeekProfileError("profile detector is not zeek-profile")
        profile = cls(
            schema_version=SCHEMA_VERSION,
            product_version=product_version,
            sensor=sensor,
            feature_names=PROFILE_FEATURES,
            center=center,
            scale=scale,
            fit_records=fit_records,
            threshold=threshold,
            mad=mad,
        )
        if payload.get("profile_fingerprint") != profile.fingerprint:
            raise ZeekProfileError("profile fingerprint does not match its contents")
        if payload.get("mad_sha256") != profile.mad_fingerprint:
            raise ZeekProfileError("profile MAD fingerprint does not match its contents")
        return profile


@dataclass(frozen=True)
class ZeekScoreResult:
    """Score output plus explicit accounting for skipped input rows."""

    findings: tuple[dict[str, Any], ...]
    parsed_records: int
    scored_records: int
    anomalous_records: int
    excluded_records: int
    incomplete_records: int
    invalid_records: int
    duplicate_records: int


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ZeekInputError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ZeekInputError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ZeekInputError(f"{field} must be finite")
    return number


def _finite_mapping(value: Any, field: str, *, positive: bool) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ZeekProfileError(f"profile {field} must be an object")
    result: dict[str, float] = {}
    for name in PROFILE_FEATURES:
        if name not in value:
            raise ZeekProfileError(f"profile {field} is missing {name}")
        number = _finite_number(value[name], f"profile {field}.{name}")
        if positive and number <= 0:
            raise ZeekProfileError(f"profile {field}.{name} must be positive")
        result[name] = number
    return result


def _validate_sensor(value: Any) -> str:
    if not isinstance(value, str):
        raise ZeekInputError("sensor must be a string")
    sensor = value.strip()
    if not _SENSOR_RE.fullmatch(sensor):
        raise ZeekInputError(
            "sensor must start with an alphanumeric character and contain only "
            "letters, digits, '.', ':', '/', '-', or '_'"
        )
    return sensor


def _parse_epoch(value: Any) -> datetime:
    timestamp = _finite_number(value, "ts")
    try:
        result = datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise ZeekInputError("ts is outside the supported UTC range") from error
    return result


def _parse_ip(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ZeekInputError(f"{field} must be a non-empty IP address")
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError as error:
        raise ZeekInputError(f"{field} is not a valid IP address") from error


def _parse_port(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ZeekInputError(f"{field} must be an integer in [0, 65535]")
    if not 0 <= value <= 65535:
        raise ZeekInputError(f"{field} must be an integer in [0, 65535]")
    return value


def _parse_record(payload: Mapping[str, Any]) -> ZeekRecord:
    missing = [field for field in _REQUIRED_FIELDS if field not in payload or payload[field] is None]
    if missing:
        raise ZeekIncompleteRecord(
            f"incomplete Zeek record; missing {', '.join(missing)}"
        )
    uid = payload["uid"]
    if not isinstance(uid, str) or not uid.strip():
        raise ZeekInputError("uid must be a non-empty string")
    protocol = payload["proto"]
    if not isinstance(protocol, str) or not protocol.strip():
        raise ZeekInputError("proto must be a non-empty string")

    features: dict[str, float] = {}
    missing_features: list[str] = []
    for name in PROFILE_FEATURES:
        value = payload.get(name)
        if value is None or value == "":
            missing_features.append(name)
            continue
        number = _finite_number(value, name)
        if name in _INTEGER_FEATURES and type(value) is not int:
            raise ZeekInputError(f"{name} must be an integer")
        if number < 0:
            raise ZeekInputError(f"{name} must be non-negative")
        features[name] = number

    return ZeekRecord(
        uid=uid.strip(),
        timestamp=_parse_epoch(payload["ts"]),
        src_ip=_parse_ip(payload["id.orig_h"], "id.orig_h"),
        src_port=_parse_port(payload["id.orig_p"], "id.orig_p"),
        dst_ip=_parse_ip(payload["id.resp_h"], "id.resp_h"),
        dst_port=_parse_port(payload["id.resp_p"], "id.resp_p"),
        protocol=protocol.strip().lower(),
        features=features,
        missing_features=tuple(missing_features),
    )


def read_zeek_jsonl(path: str | Path) -> ZeekBatch:
    """Read Zeek JSONL, retaining metadata-valid incomplete rows for accounting.

    Invalid JSON, invalid metadata, and duplicate UIDs are excluded.  A row
    with valid connection metadata but a missing profile field is retained as
    incomplete and is never imputed or scored.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise ZeekInputError(f"Zeek JSONL input not found: {source}")
    records: list[ZeekRecord] = []
    seen_uids: set[str] = set()
    lines_read = 0
    excluded_records = 0
    incomplete_records = 0
    invalid_records = 0
    duplicate_records = 0
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as error:
        raise ZeekInputError(f"cannot read Zeek JSONL input {source}: {error}") from error

    def _lines():
        try:
            yield from handle
        except UnicodeError as error:
            raise ZeekInputError(f"Zeek JSONL input is not valid UTF-8: {error}") from error

    with handle:
        for _line_number, line in enumerate(_lines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines_read += 1
            try:
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ZeekInputError("JSON record must be an object")
                record = _parse_record(payload)
            except ZeekIncompleteRecord:
                excluded_records += 1
                incomplete_records += 1
                continue
            except (json.JSONDecodeError, ZeekInputError):
                excluded_records += 1
                invalid_records += 1
                continue
            if record.uid in seen_uids:
                excluded_records += 1
                duplicate_records += 1
                continue
            seen_uids.add(record.uid)
            records.append(record)
            if not record.complete:
                incomplete_records += 1
    parsed_records = len(records)
    scorable_records = sum(record.complete for record in records)
    return ZeekBatch(
        records=tuple(records),
        stats=ZeekParseStats(
            lines_read=lines_read,
            parsed_records=parsed_records,
            scorable_records=scorable_records,
            excluded_records=excluded_records,
            incomplete_records=incomplete_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_records,
        ),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


def fit_profile(
    batch: ZeekBatch,
    *,
    sensor: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_records: int = MIN_DEFAULT_RECORDS,
) -> ZeekProfile:
    """Fit a robust Zeek-native profile from complete records only."""

    sensor = _validate_sensor(sensor)
    threshold = _finite_number(threshold, "threshold")
    if threshold < 0:
        raise ZeekProfileError("threshold must be non-negative")
    if isinstance(min_records, bool) or not isinstance(min_records, int) or min_records < 1:
        raise ZeekProfileError("min_records must be a positive integer")
    complete = batch.complete_records
    if len(complete) < min_records:
        raise ZeekProfileError(
            f"need at least {min_records} complete Zeek records to fit; got {len(complete)}"
        )
    center: dict[str, float] = {}
    mad_values: dict[str, float] = {}
    scale: dict[str, float] = {}
    for name in PROFILE_FEATURES:
        values = [record.features[name] for record in complete]
        median = _median(values)
        raw_mad = _median([abs(value - median) for value in values])
        # A zero MAD is meaningful for a stable sensor.  Keep a small positive
        # denominator so a departure from a constant baseline is visible,
        # without inventing a replacement value for the measured field.
        center[name] = median
        mad_values[name] = raw_mad
        scale[name] = max(MAD_SCALE_FACTOR * raw_mad, MAD_SCALE_FLOOR)
    return ZeekProfile(
        schema_version=SCHEMA_VERSION,
        product_version=__version__,
        sensor=sensor,
        feature_names=PROFILE_FEATURES,
        center=center,
        scale=scale,
        fit_records=len(complete),
        threshold=threshold,
        mad=mad_values,
    )


def _write_profile(path: str | Path, profile: ZeekProfile) -> None:
    destination = Path(path).expanduser()
    with atomic_text_output(destination) as handle:
        json.dump(profile.to_dict(), handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def load_profile(path: str | Path) -> ZeekProfile:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ZeekProfileError(f"Zeek profile not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ZeekProfileError(f"cannot read Zeek profile {source}: {error}") from error
    return ZeekProfile.from_dict(payload)


def _score_value(record: ZeekRecord, profile: ZeekProfile) -> tuple[float, dict[str, float]]:
    deviations = {
        name: abs(record.features[name] - profile.center[name]) / profile.scale[name]
        for name in profile.feature_names
    }
    return max(deviations.values()), deviations


def _mad_explanation_values(
    profile: ZeekProfile, name: str
) -> tuple[float | None, float | None, bool, str]:
    """Return raw MAD, pre-floor scale, floor flag and provenance status.

    New profiles persist the raw MAD so an exactly constant feature can be
    distinguished from a feature whose non-zero MAD is smaller than the
    numerical denominator floor.  Older profile files contain only the
    already-floored scale.  Their non-floor values can be reconstructed, while
    a floor-sized value is reported as unknown instead of being mislabelled.
    """

    if profile.mad is not None and name in profile.mad:
        raw_mad = float(profile.mad[name])
        raw_scale = MAD_SCALE_FACTOR * raw_mad
        return raw_mad, raw_scale, raw_scale < MAD_SCALE_FLOOR, (
            "zero"
            if raw_mad == 0.0
            else "tiny_nonzero"
            if raw_scale < MAD_SCALE_FLOOR
            else "nonzero"
        )
    profile_scale = float(profile.scale[name])
    if profile_scale > MAD_SCALE_FLOOR:
        return (
            profile_scale / MAD_SCALE_FACTOR,
            profile_scale,
            False,
            "derived_from_legacy_scale",
        )
    return None, None, True, "unknown_legacy_floor"


def explain_score(
    record: ZeekRecord,
    profile: ZeekProfile,
    *,
    score: float | None = None,
    deviations: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Explain the strongest robust-deviation dimensions for one record.

    The returned tuple contains every feature tied for the maximum score in
    profile order.  It is intentionally independent of the alert threshold so
    queue consumers can use the same witness before deciding whether to emit a
    finding.  ``score`` and ``deviations`` may be supplied by a caller that
    already computed them, avoiding a second calculation while preserving the
    exact score values.
    """

    if not record.complete:
        raise ZeekInputError("cannot explain an incomplete Zeek record")
    if deviations is None:
        calculated_score, calculated_deviations = _score_value(record, profile)
        deviations = calculated_deviations
    else:
        missing = set(profile.feature_names) - set(deviations)
        if missing:
            raise ZeekInputError(
                f"deviations are missing profile features: {', '.join(sorted(missing))}"
            )
        calculated_score = max(float(deviations[name]) for name in profile.feature_names)
    effective_score = calculated_score if score is None else float(score)
    strongest = [
        name
        for name in profile.feature_names
        if float(deviations[name]) == calculated_score
    ]
    explanations: list[dict[str, Any]] = []
    for name in strongest:
        observed = float(record.features[name])
        reference_median = float(profile.center[name])
        absolute_delta = abs(observed - reference_median)
        reference_mad, scale_before_floor, floor_applied, mad_status = _mad_explanation_values(
            profile, name
        )
        floor_note = (
            f", floor {MAD_SCALE_FLOOR:g} applied"
            if floor_applied
            else ""
        )
        mad_note = (
            "unknown (legacy profile at floor)"
            if reference_mad is None
            else f"{reference_mad:g} ({mad_status})"
        )
        reason = (
            f"{name}: observed {observed:g} vs reference median "
            f"{reference_median:g}, absolute delta {absolute_delta:g}, "
            f"reference MAD {mad_note}, robust scale {profile.scale[name]:g}"
            f"{floor_note}, deviation {float(deviations[name]):.6g}"
        )
        explanations.append(
            {
                "feature": name,
                "observed": observed,
                "reference_median": reference_median,
                "absolute_delta": absolute_delta,
                "reference_mad": reference_mad,
                "mad_status": mad_status,
                "robust_scale": float(profile.scale[name]),
                "scale_before_floor": scale_before_floor,
                "scale_floor": MAD_SCALE_FLOOR,
                "scale_floor_applied": floor_applied,
                "deviation": float(deviations[name]),
                "score": effective_score,
                "reason": reason,
            }
        )
    return tuple(explanations)


def _finding_for_record(
    record: ZeekRecord,
    profile: ZeekProfile,
    *,
    score: float,
    deviations: Mapping[str, float],
    threshold: float,
) -> dict[str, Any]:
    anomalous_features = [
        name for name in profile.feature_names if deviations[name] >= threshold
    ]
    score_explanation = explain_score(
        record,
        profile,
        score=score,
        deviations=deviations,
    )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "sensor": profile.sensor,
        "profile_fingerprint": profile.fingerprint,
        "protocol": record.protocol,
        "measured": {name: record.features[name] for name in profile.feature_names},
        "training_baseline": {
            "algorithm": "robust-median-mad",
            "fit_records": profile.fit_records,
            "center": dict(profile.center),
            "scale": dict(profile.scale),
            "profile_fingerprint": profile.fingerprint,
        },
        "deviations": dict(deviations),
        "score": score,
        "threshold": threshold,
        "anomalous_features": anomalous_features,
        "score_explanation": list(score_explanation),
        "imputation": "none",
    }
    # The shared builder scopes its stable fingerprint by sensor.  Keep the
    # original Zeek uid as the observation id so operators can pivot directly
    # to conn.log without reverse-mapping a synthetic identifier.
    return detection_finding(
        sensor=profile.sensor,
        observation_id=record.uid,
        first_seen=record.timestamp,
        last_seen=record.end_time,
        src_ip=record.src_ip,
        dst_ip=record.dst_ip,
        dst_port=record.dst_port,
        src_port=record.src_port,
        detector=DETECTOR_NAME,
        detector_version=profile.schema_version,
        title="Zeek connection deviates from the sensor baseline",
        evidence=evidence,
        product_version=__version__,
    )


def score_batch(
    batch: ZeekBatch,
    profile: ZeekProfile,
    *,
    sensor: str | None = None,
    threshold: float | None = None,
) -> ZeekScoreResult:
    """Score complete records and build OCSF-compatible findings for anomalies."""

    if sensor is not None and _validate_sensor(sensor) != profile.sensor:
        raise ZeekProfileError(
            f"profile sensor {profile.sensor!r} does not match requested sensor {sensor!r}"
        )
    effective_threshold = profile.threshold if threshold is None else _finite_number(threshold, "threshold")
    if effective_threshold < 0:
        raise ZeekProfileError("threshold must be non-negative")
    findings: list[dict[str, Any]] = []
    for record in batch.complete_records:
        score, deviations = _score_value(record, profile)
        if score < effective_threshold:
            continue
        findings.append(
            _finding_for_record(
                record,
                profile,
                score=score,
                deviations=deviations,
                threshold=effective_threshold,
            )
        )
    return ZeekScoreResult(
        findings=tuple(findings),
        parsed_records=batch.stats.parsed_records,
        scored_records=len(batch.complete_records),
        anomalous_records=len(findings),
        excluded_records=batch.stats.excluded_records,
        incomplete_records=batch.stats.incomplete_records,
        invalid_records=batch.stats.invalid_records,
        duplicate_records=batch.stats.duplicate_records,
    )


def _command_fit(args: argparse.Namespace) -> int:
    batch = read_zeek_jsonl(args.input)
    output = Path(args.output).expanduser().resolve()
    source = Path(args.input).expanduser().resolve()
    ensure_distinct(output, source)
    profile = fit_profile(
        batch,
        sensor=args.sensor,
        threshold=args.threshold,
        min_records=args.min_records,
    )
    _write_profile(output, profile)
    print(
        json.dumps(
            {
                "command": "zeek-fit",
                "sensor": profile.sensor,
                "fit_records": profile.fit_records,
                "excluded_records": batch.stats.excluded_records,
                "incomplete_records": batch.stats.incomplete_records,
                "invalid_records": batch.stats.invalid_records,
                "duplicate_records": batch.stats.duplicate_records,
                "profile": str(output),
                "schema_version": profile.schema_version,
            },
            allow_nan=False,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    return 0


def _command_score(args: argparse.Namespace) -> int:
    batch = read_zeek_jsonl(args.input)
    source = Path(args.input).expanduser().resolve()
    profile_path = Path(args.profile).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    profile = load_profile(profile_path)
    result = score_batch(
        batch,
        profile,
        sensor=args.sensor,
        threshold=args.threshold,
    )
    ensure_distinct(output, source, profile_path)
    write_incidents_jsonl(output, result.findings)
    print(
        json.dumps(
            {
                "command": "zeek-score",
                "sensor": profile.sensor,
                "parsed_records": result.parsed_records,
                "scored_records": result.scored_records,
                "anomalous_records": result.anomalous_records,
                "excluded_records": result.excluded_records,
                "incomplete_records": result.incomplete_records,
                "invalid_records": result.invalid_records,
                "duplicate_records": result.duplicate_records,
                "threshold": profile.threshold if args.threshold is None else args.threshold,
                "output": str(output),
            },
            allow_nan=False,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    return 0


def register_parser(subparsers: Any) -> None:
    """Register ``zeek-fit`` and ``zeek-score`` with the root CLI parser."""

    fit = subparsers.add_parser(
        "zeek-fit",
        help="fit a sensor-local robust baseline from Zeek conn.log JSONL",
    )
    fit.add_argument("input", type=Path, help="Zeek JSONL conn.log used for fitting")
    fit.add_argument("--sensor", required=True, help="stable sensor identifier")
    fit.add_argument("--output", required=True, type=Path, help="profile JSON path")
    fit.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="robust-deviation alert threshold (default: %(default)s)",
    )
    fit.add_argument(
        "--min-records",
        type=int,
        default=MIN_DEFAULT_RECORDS,
        help="minimum complete records required for fitting (default: %(default)s)",
    )
    fit.set_defaults(handler=_command_fit)

    score = subparsers.add_parser(
        "zeek-score",
        help="score held-out Zeek conn.log JSONL against a local profile",
    )
    score.add_argument("input", type=Path, help="held-out Zeek JSONL conn.log")
    score.add_argument("--profile", required=True, type=Path, help="profile JSON path")
    score.add_argument("--sensor", required=True, help="sensor identifier matching the profile")
    score.add_argument("--output", required=True, type=Path, help="OCSF-compatible JSONL path")
    score.add_argument(
        "--threshold",
        type=float,
        help="optional score threshold override; defaults to the fitted profile",
    )
    score.set_defaults(handler=_command_score)


__all__ = [
    "DEFAULT_THRESHOLD",
    "DETECTOR_NAME",
    "MAD_SCALE_FACTOR",
    "MAD_SCALE_FLOOR",
    "MIN_DEFAULT_RECORDS",
    "PROFILE_FEATURES",
    "SCHEMA_VERSION",
    "ZeekBatch",
    "ZeekError",
    "ZeekIncompleteRecord",
    "ZeekInputError",
    "ZeekParseStats",
    "ZeekProfile",
    "ZeekProfileError",
    "ZeekRecord",
    "ZeekScoreResult",
    "fit_profile",
    "explain_score",
    "load_profile",
    "read_zeek_jsonl",
    "register_parser",
    "score_batch",
]
