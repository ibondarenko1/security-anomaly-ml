"""Shared deterministic incident-v1 mapping and UTF-8 JSONL serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .results import BatchAnalysisResult, IncidentDetection


INCIDENT_SCHEMA_VERSION = "incident-v1"
PRODUCT_VERSION = "0.1.0"
MODEL_VERSION = "context-rf-v2"
FEATURE_CONTRACT_VERSION = "cicflow-v2-128"


class IncidentSerializationError(ValueError):
    """Raised when an internal incident cannot satisfy incident-v1."""


class OutputWriteError(OSError):
    """Raised when an atomic public output cannot be completed."""


def _source_naive_iso(value: datetime) -> str:
    if value.utcoffset() is not None:
        raise IncidentSerializationError(
            "incident-v1 timestamps must retain source-defined naive timezone semantics"
        )
    return value.isoformat(timespec="seconds")


def incident_id_v1(incident: IncidentDetection) -> str:
    """Return the stable content-derived public identifier for one incident."""

    first_seen = _source_naive_iso(incident.first_seen)
    last_seen = _source_naive_iso(incident.last_seen)
    canonical = (
        f"incident-v1|src_ip={incident.src_ip}|dst_ip={incident.dst_ip}"
        f"|dst_port={int(incident.dst_port)}|first_seen={first_seen}"
        f"|last_seen={last_seen}"
    )
    return "inc_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def incident_to_v1(
    incident: IncidentDetection,
    *,
    product_version: str,
    model_version: str,
    feature_contract: str,
) -> dict[str, Any]:
    """Map one promoted internal incident to the fail-closed public contract."""

    if not incident.promoted:
        raise IncidentSerializationError("Public incident-v1 output is promoted-only")
    if (product_version, model_version, feature_contract) != (
        PRODUCT_VERSION,
        MODEL_VERSION,
        FEATURE_CONTRACT_VERSION,
    ):
        raise IncidentSerializationError(
            "Incident metadata does not match the frozen v0.1 public contract"
        )
    if not str(incident.src_ip).strip() or not str(incident.dst_ip).strip():
        raise IncidentSerializationError("Incident source and destination IP must be non-empty")
    if incident.last_seen < incident.first_seen:
        raise IncidentSerializationError("Incident last_seen cannot precede first_seen")
    protocols = sorted({int(value) for value in incident.protocols})
    if not protocols or any(value < 0 or value > 255 for value in protocols):
        raise IncidentSerializationError("protocols must contain integers in [0, 255]")
    if incident.dst_port < 0 or incident.dst_port > 65535:
        raise IncidentSerializationError("dst_port must be in [0, 65535]")
    if incident.flow_count < 1:
        raise IncidentSerializationError("flow_count must be positive")
    scores = (float(incident.max_attack_score), float(incident.mean_attack_score))
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in scores):
        raise IncidentSerializationError("Incident scores must be finite and in [0, 1]")

    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_id": incident_id_v1(incident),
        "first_seen": _source_naive_iso(incident.first_seen),
        "last_seen": _source_naive_iso(incident.last_seen),
        "src_ip": str(incident.src_ip),
        "dst_ip": str(incident.dst_ip),
        "dst_port": int(incident.dst_port),
        "protocols": protocols,
        "flow_count": int(incident.flow_count),
        "max_attack_score": scores[0],
        "mean_attack_score": scores[1],
        "promoted": True,
        "product_version": product_version,
        "model_version": model_version,
        "feature_contract": feature_contract,
    }


def promoted_incidents_to_v1(result: BatchAnalysisResult) -> tuple[dict[str, Any], ...]:
    """Convert exactly the detector's promoted incidents using one shared mapping."""

    return tuple(
        incident_to_v1(
            incident,
            product_version=result.product_version,
            model_version=result.model_version,
            feature_contract=result.feature_contract,
        )
        for incident in result.promoted_incidents
    )


def jsonl_text(incidents: Iterable[Mapping[str, Any]]) -> str:
    """Serialize deterministic strict JSON Lines; an empty result is an empty file."""

    lines = [
        json.dumps(
            dict(incident),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for incident in incidents
    ]
    return "" if not lines else "\n".join(lines) + "\n"


def write_incidents_jsonl(
    path: str | Path,
    incidents: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write strict UTF-8 JSONL through a sibling temporary file."""

    destination = Path(path)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(jsonl_text(incidents))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, TypeError, ValueError) as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(error, IncidentSerializationError):
            raise
        raise OutputWriteError(f"Cannot atomically write {destination}: {error}") from error
