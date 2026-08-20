from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.security_anomaly.aggregation import aggregate_policy_b_alerts
from src.security_anomaly.results import FlowDetection
from tools.verify_product_incident_parity import (
    compare_incident_tables,
    product_incident_table,
    reference_incident_table,
)


def detection(row: int, second: int, score: float, protocol: int = 6):
    return FlowDetection(
        source_row_id=row,
        timestamp=datetime(2015, 2, 17, 20, 0, second),
        flow_id=str(row),
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=50000 + row,
        dst_port=80,
        protocol=protocol,
        attack_score=score,
        is_alert=score >= 0.10,
    )


def test_reference_comparison_proves_membership_attributes_and_promotion() -> None:
    flows = (
        detection(0, 0, 0.20, protocol=6),
        detection(1, 10, 0.25, protocol=17),
        detection(2, 20, 0.01),
    )
    product_incidents = aggregate_policy_b_alerts(
        flows, window_seconds=300, promotion_threshold=0.25
    )
    source_to_validation = np.array([2, 0, 1], dtype=np.int64)
    product = product_incident_table(product_incidents, source_to_validation)
    reference = pd.DataFrame(
        {
            "validation_row": [2, 0, 1],
            "timestamp": pd.to_datetime(
                [flow.timestamp for flow in flows]
            ),
            "src_ip": [flow.src_ip for flow in flows],
            "dst_ip": [flow.dst_ip for flow in flows],
            "dst_port": [flow.dst_port for flow in flows],
            "attack_score": [flow.attack_score for flow in flows],
            "is_alert": [flow.is_alert for flow in flows],
        }
    )
    expected = reference_incident_table(reference)

    report = compare_incident_tables(product, expected)

    assert report["exact_membership_parity"] is True
    assert report["missing_reference_memberships"] == 0
    assert report["extra_product_memberships"] == 0
    assert report["grouping_key_disagreements"] == 0
    assert report["promotion_decision_disagreements"] == 0


def test_comparison_detects_membership_disagreement() -> None:
    reference = pd.DataFrame(
        {
            "validation_rows": [(0, 1)],
            "first_seen": [pd.Timestamp("2015-02-17 20:00:00")],
            "last_seen": [pd.Timestamp("2015-02-17 20:00:01")],
            "src_ip": ["a"],
            "dst_ip": ["b"],
            "dst_port": [80],
            "flow_count": [2],
            "max_attack_score": [0.3],
            "mean_attack_score": [0.25],
            "promoted": [True],
        }
    )
    product = reference.copy()
    product["validation_rows"] = [(0, 2)]
    report = compare_incident_tables(product, reference)
    assert report["exact_membership_parity"] is False
    assert report["missing_reference_memberships"] == 1
    assert report["extra_product_memberships"] == 1
