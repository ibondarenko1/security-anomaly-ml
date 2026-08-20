from __future__ import annotations

from datetime import datetime, timedelta

from src.security_anomaly.aggregation import aggregate_policy_b_alerts
from src.security_anomaly.results import FlowDetection


START = datetime(2015, 2, 17, 20, 0, 0)


def flow(
    source_row_id: int,
    seconds: float,
    score: float,
    *,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    dst_port: int = 80,
    protocol: int = 6,
    is_alert: bool | None = None,
) -> FlowDetection:
    return FlowDetection(
        source_row_id=source_row_id,
        timestamp=START + timedelta(seconds=seconds),
        flow_id=f"flow-{source_row_id}",
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=50000 + source_row_id,
        dst_port=dst_port,
        protocol=protocol,
        attack_score=score,
        is_alert=score >= 0.10 if is_alert is None else is_alert,
    )


def aggregate(*flows: FlowDetection):
    return aggregate_policy_b_alerts(
        flows, window_seconds=300, promotion_threshold=0.25
    )


def test_zero_alerts_produce_no_incidents() -> None:
    assert aggregate(flow(0, 0, 0.09), flow(1, 10, 0.01)) == ()


def test_one_alert_produces_one_incident() -> None:
    incidents = aggregate(flow(0, 0, 0.10))
    assert len(incidents) == 1
    assert incidents[0].source_row_ids == (0,)
    assert incidents[0].flow_count == 1


def test_gap_equal_to_300_seconds_remains_in_same_incident() -> None:
    incidents = aggregate(flow(0, 0, 0.20), flow(1, 300, 0.21))
    assert len(incidents) == 1
    assert incidents[0].source_row_ids == (0, 1)


def test_gap_greater_than_300_seconds_starts_new_incident() -> None:
    incidents = aggregate(flow(0, 0, 0.20), flow(1, 300.000001, 0.21))
    assert len(incidents) == 2
    assert [incident.source_row_ids for incident in incidents] == [(0,), (1,)]


def test_non_alert_flow_does_not_bridge_alert_sessions() -> None:
    incidents = aggregate(
        flow(0, 0, 0.20),
        flow(1, 200, 0.01),
        flow(2, 400, 0.20),
    )
    assert len(incidents) == 2
    assert [incident.source_row_ids for incident in incidents] == [(0,), (2,)]


def test_protocol_is_evidence_not_part_of_policy_b_key() -> None:
    incidents = aggregate(
        flow(0, 0, 0.20, protocol=6),
        flow(1, 10, 0.20, protocol=17),
    )
    assert len(incidents) == 1
    assert incidents[0].protocols == (6, 17)


def test_multiple_keys_and_sessions_have_deterministic_order() -> None:
    inputs = (
        flow(4, 700, 0.20),
        flow(2, 10, 0.20, dst_port=443),
        flow(0, 0, 0.20),
        flow(3, 400, 0.20),
        flow(1, 0, 0.20),
    )
    first = aggregate(*inputs)
    second = aggregate(*reversed(inputs))

    def operational_view(incidents):
        return [
            (
                item.incident_sequence,
                item.first_seen,
                item.src_ip,
                item.dst_ip,
                item.dst_port,
                tuple(sorted(item.source_row_ids)),
            )
            for item in incidents
        ]

    assert operational_view(first) == operational_view(second)
    assert [incident.incident_sequence for incident in first] == list(range(len(first)))


def test_promotion_boundary_is_inclusive() -> None:
    incidents = aggregate(
        flow(0, 0, 0.249999),
        flow(1, 400, 0.25),
    )
    assert [incident.promoted for incident in incidents] == [False, True]
    assert incidents[0].mean_attack_score == 0.249999
    assert incidents[1].max_attack_score == 0.25


def test_same_timestamp_members_use_source_row_tie_breaker() -> None:
    incidents = aggregate(flow(9, 0, 0.20), flow(3, 0, 0.21))
    assert incidents[0].source_row_ids == (3, 9)
