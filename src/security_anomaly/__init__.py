"""Production-oriented inference primitives for Security Anomaly ML v0.1."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ArtifactIntegrityError",
    "CausalTemporalFeatureBuilder",
    "ContractError",
    "FeatureBatch",
    "FeatureContract",
    "FrozenModelBundle",
    "InputContractError",
    "ModelCompatibilityError",
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
