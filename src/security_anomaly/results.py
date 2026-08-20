"""Internal Python result contracts for the frozen v0.1 detector core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FlowDetection:
    """One temporally ordered flow and its frozen attack-score decision."""

    source_row_id: int
    timestamp: datetime
    flow_id: str | None
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    attack_score: float
    is_alert: bool


@dataclass(frozen=True)
class IncidentDetection:
    """One Policy B rolling inactivity-gap incident."""

    incident_sequence: int
    first_seen: datetime
    last_seen: datetime
    src_ip: str
    dst_ip: str
    dst_port: int
    protocols: tuple[int, ...]
    flow_count: int
    max_attack_score: float
    mean_attack_score: float
    promoted: bool
    source_row_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchAnalysisResult:
    """Structured in-process result; this is not the public incident-v1 schema."""

    flows_processed: int
    flow_alert_count: int
    incident_count: int
    promoted_incident_count: int
    flow_detections: tuple[FlowDetection, ...]
    incidents: tuple[IncidentDetection, ...]
    promoted_incidents: tuple[IncidentDetection, ...]
    product_version: str
    model_version: str
    feature_contract: str
    state_mode: str
