from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.security_anomaly.contracts import FeatureContract
from src.security_anomaly.detector import (
    FrozenOperationalPolicy,
    SecurityAnomalyDetector,
)
from src.security_anomaly.model_bundle import ModelCompatibilityError
from src.security_anomaly.temporal import CausalTemporalFeatureBuilder, FeatureBatch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "contracts" / "model-manifest-context-rf-v2.json"


class FakeBundle:
    def __init__(self, contract: FeatureContract, scores: list[float]) -> None:
        self.contract = contract
        self.model_version = "context-rf-v2"
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self._scores = np.asarray(scores, dtype=float)

    def predict_scores(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape == (len(self._scores), 128)
        return self._scores.copy()


class FakeBuilder:
    def __init__(self, contract: FeatureContract) -> None:
        self.contract = contract
        self.calls = 0

    def build(self, records) -> FeatureBatch:
        self.calls += 1
        identities = pd.DataFrame(
            {
                "source_row_id": [1, 0, 2],
                "timestamp": pd.to_datetime(
                    [
                        "2015-02-17 20:00:00",
                        "2015-02-17 20:00:00",
                        "2015-02-17 20:06:00",
                    ]
                ),
                "flow_id": ["second", "first", "third"],
                "src_ip": ["10.0.0.1"] * 3,
                "dst_ip": ["10.0.0.2"] * 3,
                "src_port": [50001, 50000, 50002],
                "dst_port": [80, 80, 80],
                "protocol": [17, 6, 6],
            }
        )
        features = pd.DataFrame(
            np.zeros((3, 128), dtype=np.float32),
            columns=self.contract.model_features,
        )
        return FeatureBatch(identities, features, self.contract.version)


class ConstantFakeBundle(FakeBundle):
    def predict_scores(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(len(matrix), 0.20, dtype=float)


def make_detector(scores: list[float] | None = None):
    contract = FeatureContract.load()
    bundle = FakeBundle(contract, scores or [0.20, 0.30, 0.40])
    builder = FakeBuilder(contract)
    policy = FrozenOperationalPolicy.from_bundle(bundle)  # type: ignore[arg-type]
    return SecurityAnomalyDetector(bundle, builder, policy), builder  # type: ignore[arg-type]


def test_detector_composes_builder_scoring_aggregation_and_promotion() -> None:
    detector, builder = make_detector()
    result = detector.analyze_batch([{"unlabeled": True}] * 3)

    assert builder.calls == 1
    assert result.flows_processed == 3
    assert result.flow_alert_count == 3
    assert result.incident_count == 2
    assert result.promoted_incident_count == 2
    assert [flow.source_row_id for flow in result.flow_detections] == [1, 0, 2]
    assert result.incidents[0].source_row_ids == (0, 1)
    assert result.incidents[0].protocols == (6, 17)
    assert result.state_mode == "batch-empty"


def test_empty_input_is_explicit_and_does_not_call_builder() -> None:
    detector, builder = make_detector([])
    result = detector.analyze_batch([])
    assert result.flows_processed == 0
    assert result.flow_detections == ()
    assert result.incidents == ()
    assert builder.calls == 0


def test_no_label_or_attack_category_is_required() -> None:
    detector, _ = make_detector()
    result = detector.analyze_batch([{"network_flow": index} for index in range(3)])
    assert result.flows_processed == 3


def test_identical_input_is_deterministic() -> None:
    detector, _ = make_detector()
    records = [{"network_flow": index} for index in range(3)]
    assert detector.analyze_batch(records) == detector.analyze_batch(records)


def test_real_builder_makes_out_of_order_input_operationally_equivalent() -> None:
    contract = FeatureContract.load()
    rows = 4
    data: dict[str, list[object]] = {
        feature: [0.0] * rows for feature in contract.baseline_features
    }
    data.update(
        {
            "Flow ID": ["a", "b", "c", "d"],
            "Src IP": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.9"],
            "Src Port": [50000, 50001, 50002, 50003],
            "Dst IP": ["10.0.0.2", "10.0.0.2", "10.0.0.2", "10.0.0.8"],
            "Dst Port": [80, 80, 80, 53],
            "Protocol": [6, 17, 6, 17],
            "Timestamp": [
                "17/02/2015 08:00:00 PM",
                "17/02/2015 08:01:00 PM",
                "17/02/2015 08:07:00 PM",
                "17/02/2015 08:00:30 PM",
            ],
        }
    )
    frame = pd.DataFrame(data)
    bundle = ConstantFakeBundle(contract, [])
    detector = SecurityAnomalyDetector(
        bundle,  # type: ignore[arg-type]
        CausalTemporalFeatureBuilder(contract),
        FrozenOperationalPolicy.from_bundle(bundle),  # type: ignore[arg-type]
    )
    shuffled = frame.iloc[[2, 0, 3, 1]].reset_index(drop=True)

    first = detector.analyze_batch(frame)
    second = detector.analyze_batch(shuffled)

    def operational_view(result):
        flow_id_by_source = {
            flow.source_row_id: flow.flow_id for flow in result.flow_detections
        }
        return [
            (
                incident.first_seen,
                incident.last_seen,
                incident.src_ip,
                incident.dst_ip,
                incident.dst_port,
                tuple(
                    sorted(flow_id_by_source[row] for row in incident.source_row_ids)
                ),
                incident.promoted,
            )
            for incident in result.incidents
        ]

    assert operational_view(first) == operational_view(second)
    assert all(flow.attack_score == 0.20 for flow in first.flow_detections)
    assert "Label" not in frame and "attack_cat" not in frame


def test_manifest_policy_mismatch_fails_closed() -> None:
    contract = FeatureContract.load()
    bundle = FakeBundle(contract, [0.2])
    bundle.manifest = deepcopy(bundle.manifest)
    bundle.manifest["operational_policy"]["flow_alert"]["threshold"] = 0.11
    with pytest.raises(ModelCompatibilityError, match="Unsupported operational policy"):
        FrozenOperationalPolicy.from_bundle(bundle)  # type: ignore[arg-type]


def test_product_runtime_does_not_import_research_or_evaluation_modules() -> None:
    product_files = [
        ROOT / "src" / "security_anomaly" / name
        for name in ("detector.py", "aggregation.py", "results.py")
    ]
    forbidden_fragments = (
        "build_temporal_features",
        "train_",
        "select_threshold",
        "evaluate_",
        "analyze_incident",
        "attack_cat",
    )
    for path in product_files:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert not any(fragment in source for fragment in forbidden_fragments)
