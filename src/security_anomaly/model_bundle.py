"""Integrity-checked loader for the frozen Context Random Forest release asset."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .contracts import ContractError, FeatureContract, load_json, sha256_file


class ArtifactIntegrityError(RuntimeError):
    """Raised before deserialization when a release artifact hash is wrong."""


class ModelCompatibilityError(RuntimeError):
    """Raised when runtime, model, and feature-contract versions disagree."""


def _installed_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError as error:
        raise ModelCompatibilityError(f"Required runtime package is missing: {package}") from error


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


@dataclass(frozen=True)
class FrozenModelBundle:
    """Loaded model plus verified immutable release metadata."""

    model: Any
    manifest: dict[str, Any]
    contract: FeatureContract
    model_path: Path
    manifest_path: Path

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        manifest_path: str | Path,
        feature_contract_path: str | Path,
        *,
        strict_runtime: bool = True,
    ) -> "FrozenModelBundle":
        model_path = Path(model_path).resolve()
        manifest_path = Path(manifest_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Frozen model artifact not found: {model_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest not found: {manifest_path}")

        manifest = load_json(manifest_path)
        cls._validate_manifest_shape(manifest)
        artifact = manifest["artifact"]
        actual_model_hash = sha256_file(model_path)
        if actual_model_hash != artifact["sha256"]:
            raise ArtifactIntegrityError(
                "Model SHA256 mismatch before deserialization: "
                f"expected {artifact['sha256']}, got {actual_model_hash}"
            )
        if model_path.stat().st_size != artifact["size_bytes"]:
            raise ArtifactIntegrityError(
                "Model size mismatch before deserialization: "
                f"expected {artifact['size_bytes']}, got {model_path.stat().st_size}"
            )

        compatibility = manifest["compatibility"]
        try:
            contract = FeatureContract.load(
                feature_contract_path,
                expected_version=compatibility["feature_contract"],
                expected_sha256=compatibility["feature_contract_sha256"],
            )
        except ContractError as error:
            raise ModelCompatibilityError(str(error)) from error
        if contract.builder_version != compatibility["feature_builder_version"]:
            raise ModelCompatibilityError(
                "Feature-builder version mismatch: "
                f"expected {compatibility['feature_builder_version']}, "
                f"got {contract.builder_version}"
            )
        if strict_runtime:
            cls._validate_runtime(manifest["serialization"])

        # joblib/pickle loading is intentionally delayed until every available
        # integrity and compatibility check above has passed.
        model = joblib.load(model_path)
        expected_class = manifest["model"]["class"]
        actual_class = _qualified_class_name(model)
        if actual_class != expected_class:
            raise ModelCompatibilityError(
                f"Model class mismatch: expected {expected_class}, got {actual_class}"
            )
        if int(getattr(model, "n_features_in_", -1)) != len(contract.model_features):
            raise ModelCompatibilityError(
                "Model n_features_in_ does not match the 128-feature contract"
            )
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.tolist() != manifest["model"]["classes"]:
            raise ModelCompatibilityError(
                f"Model classes mismatch: expected {manifest['model']['classes']}, "
                f"got {classes.tolist()}"
            )
        actual_params = model.get_params(deep=False)
        for name, expected in manifest["model"]["frozen_parameters"].items():
            if actual_params.get(name) != expected:
                raise ModelCompatibilityError(
                    f"Frozen model parameter mismatch for {name}: "
                    f"expected {expected!r}, got {actual_params.get(name)!r}"
                )
        return cls(model, manifest, contract, model_path, manifest_path)

    @staticmethod
    def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "product_version",
            "model_version",
            "artifact",
            "serialization",
            "compatibility",
            "model",
            "operational_policy",
            "provenance",
            "redistribution",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ModelCompatibilityError(f"Model manifest is missing fields: {missing}")
        if manifest["schema_version"] != "model-bundle-manifest-v1":
            raise ModelCompatibilityError(
                f"Unsupported model manifest schema: {manifest['schema_version']}"
            )
        if manifest["model_version"] != "context-rf-v2":
            raise ModelCompatibilityError(
                f"Unsupported frozen model version: {manifest['model_version']}"
            )

    @staticmethod
    def _validate_runtime(serialization: dict[str, Any]) -> None:
        expected_python = serialization["python_major_minor"]
        actual_python = f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
        if actual_python != expected_python:
            raise ModelCompatibilityError(
                f"Python runtime mismatch: expected {expected_python}.x, got {platform.python_version()}"
            )
        mismatches = []
        for package, expected in serialization["required_package_versions"].items():
            actual = _installed_version(package)
            if actual != expected:
                mismatches.append(f"{package}: expected {expected}, got {actual}")
        if mismatches:
            raise ModelCompatibilityError(
                "Frozen model serialization runtime mismatch: " + "; ".join(mismatches)
            )

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    def metadata(self) -> dict[str, Any]:
        """Return label-free, JSON-serializable release metadata."""

        return {
            "product_version": self.manifest["product_version"],
            "model_version": self.model_version,
            "feature_contract": self.contract.version,
            "feature_builder_version": self.contract.builder_version,
            "model_sha256": self.manifest["artifact"]["sha256"],
            "model_feature_count": len(self.contract.model_features),
            "serialization": self.manifest["serialization"],
            "operational_policy": self.manifest["operational_policy"],
            "provenance": self.manifest["provenance"],
        }

    def predict_scores(self, matrix: np.ndarray) -> np.ndarray:
        """Return the frozen attack score; it is not a calibrated probability."""

        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.contract.model_features):
            raise ModelCompatibilityError(
                "Model input must have shape (rows, 128) in feature-contract order"
            )
        if not np.isfinite(values).all():
            raise ValueError("Model input contains NaN or infinity")
        scores = np.asarray(self.model.predict_proba(values)[:, 1], dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("Frozen model produced a non-finite attack score")
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Audit-only escape hatch; production loading must remain strict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FrozenModelBundle.load(
        args.model,
        args.manifest,
        args.contract,
        strict_runtime=not args.allow_runtime_mismatch,
    )
    print(json.dumps(bundle.metadata(), indent=2))


if __name__ == "__main__":
    main()
