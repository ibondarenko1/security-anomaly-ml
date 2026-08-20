"""Frozen Policy B incident aggregation for product inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
from statistics import fmean
from typing import Iterable

from .results import FlowDetection, IncidentDetection


POLICY_B_GROUPING_KEY = ("src_ip", "dst_ip", "dst_port")


def _incident_from_members(
    members: list[FlowDetection],
    *,
    promotion_threshold: float,
) -> IncidentDetection:
    if not members:
        raise ValueError("Cannot create an incident without alert flows")
    first = members[0]
    scores = [flow.attack_score for flow in members]
    maximum = max(scores)
    return IncidentDetection(
        incident_sequence=-1,
        first_seen=first.timestamp,
        last_seen=members[-1].timestamp,
        src_ip=first.src_ip,
        dst_ip=first.dst_ip,
        dst_port=first.dst_port,
        protocols=tuple(sorted({flow.protocol for flow in members})),
        flow_count=len(members),
        max_attack_score=maximum,
        mean_attack_score=fmean(scores),
        promoted=maximum >= promotion_threshold,
        source_row_ids=tuple(flow.source_row_id for flow in members),
    )


def aggregate_policy_b_alerts(
    flows: Iterable[FlowDetection],
    *,
    window_seconds: int,
    promotion_threshold: float,
) -> tuple[IncidentDetection, ...]:
    """Aggregate only alerts using consecutive-alert inactivity gaps.

    Protocol is deliberately evidence only and never participates in the key.
    A gap equal to ``window_seconds`` remains in the existing incident; only a
    strictly greater gap starts a new one.
    """

    if window_seconds <= 0:
        raise ValueError("Incident window must be positive")
    if promotion_threshold < 0 or promotion_threshold > 1:
        raise ValueError("Promotion threshold must be in [0, 1]")

    alerts_by_key: dict[tuple[str, str, int], list[FlowDetection]] = defaultdict(list)
    for flow in flows:
        if flow.is_alert:
            alerts_by_key[(flow.src_ip, flow.dst_ip, flow.dst_port)].append(flow)

    drafts: list[IncidentDetection] = []
    maximum_gap = timedelta(seconds=window_seconds)
    for members in alerts_by_key.values():
        ordered = sorted(members, key=lambda flow: (flow.timestamp, flow.source_row_id))
        session: list[FlowDetection] = []
        previous: FlowDetection | None = None
        for flow in ordered:
            if previous is not None and flow.timestamp - previous.timestamp > maximum_gap:
                drafts.append(
                    _incident_from_members(
                        session, promotion_threshold=promotion_threshold
                    )
                )
                session = []
            session.append(flow)
            previous = flow
        if session:
            drafts.append(
                _incident_from_members(session, promotion_threshold=promotion_threshold)
            )

    drafts.sort(
        key=lambda incident: (
            incident.first_seen,
            incident.src_ip,
            incident.dst_ip,
            incident.dst_port,
            incident.source_row_ids[0],
        )
    )
    return tuple(
        replace(incident, incident_sequence=sequence)
        for sequence, incident in enumerate(drafts)
    )
