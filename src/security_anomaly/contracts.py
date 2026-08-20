"""Load and validate versioned feature contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .package_resources import FEATURE_CONTRACT_RESOURCE, resource_bytes, resource_ref

# Backward-compatible name, now resolving inside the installed package rather
# than relying on PROJECT_ROOT/contracts being present.
DEFAULT_FEATURE_CONTRACT_PATH = Path(str(resource_ref(FEATURE_CONTRACT_RESOURCE)))


class ContractError(ValueError):
    """Raised when a versioned feature contract is malformed or incompatible."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


@dataclass(frozen=True)
class FeatureContract:
    """Validated ordered feature contract used by the frozen Context model."""

    path: Path | None
    sha256: str
    document: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        expected_version: str | None = None,
        expected_sha256: str | None = None,
    ) -> "FeatureContract":
        if path is None:
            raw = resource_bytes(FEATURE_CONTRACT_RESOURCE)
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ContractError(
                    "Feature contract SHA256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            document = json.loads(raw.decode("utf-8"))
            if not isinstance(document, dict):
                raise ContractError("Packaged feature contract must be a JSON object")
            contract = cls(None, actual_sha256, document)
            contract._validate(expected_version=expected_version)
            return contract
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Feature contract not found: {resolved}")
        actual_sha256 = sha256_file(resolved)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ContractError(
                "Feature contract SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        contract = cls(resolved, actual_sha256, load_json(resolved))
        contract._validate(expected_version=expected_version)
        return contract

    def _validate(self, *, expected_version: str | None) -> None:
        document = self.document
        version = document.get("feature_contract")
        if not isinstance(version, str) or not version:
            raise ContractError("feature_contract must be a non-empty string")
        if expected_version is not None and version != expected_version:
            raise ContractError(
                f"Feature contract version mismatch: expected {expected_version}, got {version}"
            )

        groups = document.get("feature_groups")
        if not isinstance(groups, dict):
            raise ContractError("feature_groups must be a JSON object")
        expected_group_counts = {"baseline": 76, "static_behavioral": 9, "temporal": 43}
        concatenated: list[str] = []
        for name, expected_count in expected_group_counts.items():
            group = groups.get(name)
            if not isinstance(group, dict) or not isinstance(group.get("features"), list):
                raise ContractError(f"feature_groups.{name}.features must be a list")
            features = group["features"]
            if group.get("count") != expected_count or len(features) != expected_count:
                raise ContractError(
                    f"feature_groups.{name} must contain exactly {expected_count} features"
                )
            if not all(isinstance(value, str) and value for value in features):
                raise ContractError(f"feature_groups.{name} contains an invalid name")
            concatenated.extend(features)

        ordered = document.get("model_features")
        if ordered != concatenated:
            raise ContractError("model_features does not match the ordered feature groups")
        if document.get("model_feature_count") != 128 or len(ordered) != 128:
            raise ContractError("The frozen model contract must contain exactly 128 features")
        if len(set(ordered)) != len(ordered):
            raise ContractError("The frozen model feature list contains duplicates")

        input_contract = document.get("input")
        if not isinstance(input_contract, dict):
            raise ContractError("input must be a JSON object")
        required_context = input_contract.get("required_identity_context_columns")
        required_model = input_contract.get("required_model_columns")
        if not isinstance(required_context, list) or not isinstance(required_model, list):
            raise ContractError("Input required-column lists are missing")
        if required_model != self.baseline_features:
            raise ContractError("Required model columns differ from baseline features")

    @property
    def version(self) -> str:
        return str(self.document["feature_contract"])

    @property
    def builder_version(self) -> str:
        return str(self.document["feature_builder_version"])

    @property
    def model_features(self) -> list[str]:
        return list(self.document["model_features"])

    @property
    def baseline_features(self) -> list[str]:
        return list(self.document["feature_groups"]["baseline"]["features"])

    @property
    def static_features(self) -> list[str]:
        return list(self.document["feature_groups"]["static_behavioral"]["features"])

    @property
    def temporal_features(self) -> list[str]:
        return list(self.document["feature_groups"]["temporal"]["features"])

    @property
    def required_input_columns(self) -> list[str]:
        input_contract = self.document["input"]
        return [
            *input_contract["required_identity_context_columns"],
            *input_contract["required_model_columns"],
        ]
