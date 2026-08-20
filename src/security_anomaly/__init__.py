"""Production-oriented inference primitives for Security Anomaly ML v0.1."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "ArtifactIntegrityError",
    "CausalTemporalFeatureBuilder",
    "ContractError",
    "BatchAnalysisResult",
    "FlowDetection",
    "FrozenOperationalPolicy",
    "FeatureBatch",
    "FeatureContract",
    "FrozenModelBundle",
    "IncidentDetection",
    "InputContractError",
    "ModelCompatibilityError",
    "SecurityAnomalyDetector",
    "ValidatedFlowBatch",
    "validate_flow_records",
    "read_and_validate_flow_csv",
    "incident_to_v1",
    "promoted_incidents_to_v1",
    "write_incidents_jsonl",
]

_EXPORTS = {
    "ContractError": (".contracts", "ContractError"),
    "FeatureContract": (".contracts", "FeatureContract"),
    "ArtifactIntegrityError": (".model_bundle", "ArtifactIntegrityError"),
    "FrozenModelBundle": (".model_bundle", "FrozenModelBundle"),
    "ModelCompatibilityError": (".model_bundle", "ModelCompatibilityError"),
    "CausalTemporalFeatureBuilder": (".temporal", "CausalTemporalFeatureBuilder"),
    "FeatureBatch": (".temporal", "FeatureBatch"),
    "InputContractError": (".temporal", "InputContractError"),
    "FlowDetection": (".results", "FlowDetection"),
    "IncidentDetection": (".results", "IncidentDetection"),
    "BatchAnalysisResult": (".results", "BatchAnalysisResult"),
    "FrozenOperationalPolicy": (".detector", "FrozenOperationalPolicy"),
    "SecurityAnomalyDetector": (".detector", "SecurityAnomalyDetector"),
    "ValidatedFlowBatch": (".validation", "ValidatedFlowBatch"),
    "validate_flow_records": (".validation", "validate_flow_records"),
    "read_and_validate_flow_csv": (".validation", "read_and_validate_flow_csv"),
    "incident_to_v1": (".serialization", "incident_to_v1"),
    "promoted_incidents_to_v1": (".serialization", "promoted_incidents_to_v1"),
    "write_incidents_jsonl": (".serialization", "write_incidents_jsonl"),
}


def __getattr__(name: str) -> Any:
    """Load public primitives lazily so ``python -m`` entry points stay clean."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
