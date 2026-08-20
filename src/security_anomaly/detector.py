"""Reusable facade for the complete frozen v0.1 operational detector path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .aggregation import POLICY_B_GROUPING_KEY, aggregate_policy_b_alerts
from .model_bundle import FrozenModelBundle, ModelCompatibilityError
from .results import BatchAnalysisResult, FlowDetection
from .temporal import CausalTemporalFeatureBuilder


EXPECTED_MODEL_VERSION = "context-rf-v2"
EXPECTED_FEATURE_CONTRACT = "cicflow-v2-128"
EXPECTED_FEATURE_BUILDER = "causal-temporal-v2"
EXPECTED_FLOW_THRESHOLD = 0.10
EXPECTED_INCIDENT_POLICY = "B"
EXPECTED_INCIDENT_WINDOW_SECONDS = 300
EXPECTED_PROMOTION_THRESHOLD = 0.25


@dataclass(frozen=True)
class FrozenOperationalPolicy:
    flow_threshold: float
    incident_policy: str
    grouping_key: tuple[str, ...]
    incident_window_seconds: int
    promotion_threshold: float

    @classmethod
    def from_bundle(cls, bundle: FrozenModelBundle) -> "FrozenOperationalPolicy":
        manifest = bundle.manifest
        if bundle.model_version != EXPECTED_MODEL_VERSION:
            raise ModelCompatibilityError(
                f"Unsupported model version: {bundle.model_version}"
            )
        if bundle.contract.version != EXPECTED_FEATURE_CONTRACT:
            raise ModelCompatibilityError(
                f"Unsupported feature contract: {bundle.contract.version}"
            )
        if bundle.contract.builder_version != EXPECTED_FEATURE_BUILDER:
            raise ModelCompatibilityError(
                f"Unsupported feature builder: {bundle.contract.builder_version}"
            )

        policy = manifest["operational_policy"]
        flow = policy["flow_alert"]
        aggregation = policy["incident_aggregation"]
        promotion = policy["incident_promotion"]
        observed = {
            "flow_field": flow.get("field"),
            "flow_operator": flow.get("operator"),
            "flow_threshold": flow.get("threshold"),
            "incident_policy": aggregation.get("policy"),
            "grouping_key": tuple(aggregation.get("grouping_key", ())),
            "incident_window_seconds": aggregation.get("window_seconds"),
            "promotion_field": promotion.get("field"),
            "promotion_operator": promotion.get("operator"),
            "promotion_threshold": promotion.get("threshold"),
        }
        expected = {
            "flow_field": "attack_score",
            "flow_operator": ">=",
            "flow_threshold": EXPECTED_FLOW_THRESHOLD,
            "incident_policy": EXPECTED_INCIDENT_POLICY,
            "grouping_key": POLICY_B_GROUPING_KEY,
            "incident_window_seconds": EXPECTED_INCIDENT_WINDOW_SECONDS,
            "promotion_field": "max_attack_score",
            "promotion_operator": ">=",
            "promotion_threshold": EXPECTED_PROMOTION_THRESHOLD,
        }
        if observed != expected:
            raise ModelCompatibilityError(
                "Unsupported operational policy in frozen manifest: "
                f"expected {expected}, got {observed}"
            )
        return cls(
            flow_threshold=float(flow["threshold"]),
            incident_policy=str(aggregation["policy"]),
            grouping_key=tuple(aggregation["grouping_key"]),
            incident_window_seconds=int(aggregation["window_seconds"]),
            promotion_threshold=float(promotion["threshold"]),
        )


class SecurityAnomalyDetector:
    """Compose validation, causal features, scoring, incidents, and promotion."""

    def __init__(
        self,
        bundle: FrozenModelBundle,
        feature_builder: CausalTemporalFeatureBuilder,
        policy: FrozenOperationalPolicy,
    ) -> None:
        if feature_builder.contract.sha256 != bundle.contract.sha256:
            raise ModelCompatibilityError(
                "Detector feature builder and model bundle use different contracts"
            )
        self.bundle = bundle
        self.feature_builder = feature_builder
        self.policy = policy

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        manifest_path: str | Path,
        feature_contract_path: str | Path,
    ) -> "SecurityAnomalyDetector":
        bundle = FrozenModelBundle.load(
            model_path=model_path,
            manifest_path=manifest_path,
            feature_contract_path=feature_contract_path,
        )
        policy = FrozenOperationalPolicy.from_bundle(bundle)
        return cls(bundle, CausalTemporalFeatureBuilder(bundle.contract), policy)

    def _empty_result(self) -> BatchAnalysisResult:
        return BatchAnalysisResult(
            flows_processed=0,
            flow_alert_count=0,
            incident_count=0,
            promoted_incident_count=0,
            flow_detections=(),
            incidents=(),
            promoted_incidents=(),
            product_version=str(self.bundle.manifest["product_version"]),
            model_version=self.bundle.model_version,
            feature_contract=self.bundle.contract.version,
            state_mode="batch-empty",
        )

    def analyze_batch(
        self, records: pd.DataFrame | Sequence[Mapping[str, Any]]
    ) -> BatchAnalysisResult:
        """Run the complete frozen path with temporal state empty at batch start."""

        if len(records) == 0:
            return self._empty_result()
        batch = self.feature_builder.build(records)
        scores = self.bundle.predict_scores(batch.to_numpy())
        detections: list[FlowDetection] = []
        for identity, score in zip(
            batch.identities.itertuples(index=False), scores, strict=True
        ):
            flow_id = None if pd.isna(identity.flow_id) else str(identity.flow_id)
            detections.append(
                FlowDetection(
                    source_row_id=int(identity.source_row_id),
                    timestamp=pd.Timestamp(identity.timestamp).to_pydatetime(),
                    flow_id=flow_id,
                    src_ip=str(identity.src_ip),
                    dst_ip=str(identity.dst_ip),
                    src_port=int(identity.src_port),
                    dst_port=int(identity.dst_port),
                    protocol=int(identity.protocol),
                    attack_score=float(score),
                    is_alert=bool(score >= self.policy.flow_threshold),
                )
            )
        flow_detections = tuple(detections)
        incidents = aggregate_policy_b_alerts(
            flow_detections,
            window_seconds=self.policy.incident_window_seconds,
            promotion_threshold=self.policy.promotion_threshold,
        )
        promoted = tuple(incident for incident in incidents if incident.promoted)
        alert_count = sum(detection.is_alert for detection in flow_detections)
        return BatchAnalysisResult(
            flows_processed=len(flow_detections),
            flow_alert_count=alert_count,
            incident_count=len(incidents),
            promoted_incident_count=len(promoted),
            flow_detections=flow_detections,
            incidents=incidents,
            promoted_incidents=promoted,
            product_version=str(self.bundle.manifest["product_version"]),
            model_version=self.bundle.model_version,
            feature_contract=self.bundle.contract.version,
            state_mode=batch.state_mode,
        )
