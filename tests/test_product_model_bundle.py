from __future__ import annotations

import hashlib
import json
import platform
from copy import deepcopy
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.security_anomaly.contracts import FeatureContract
from src.security_anomaly.model_bundle import (
    ArtifactIntegrityError,
    FrozenModelBundle,
    ModelCompatibilityError,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
FROZEN_MANIFEST_PATH = ROOT / "contracts" / "model-manifest-context-rf-v2.json"
FROZEN_MODEL_PATH = ROOT / "models" / "v2_context_random_forest.joblib"


class FakeProbabilityModel:
    def __init__(self, feature_count: int, constant: float = 0.25) -> None:
        self.n_features_in_ = feature_count
        self.classes_ = np.array([0, 1])
        self.constant = constant

    def get_params(self, deep: bool = False) -> dict[str, float]:
        return {"constant": self.constant}

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = np.full(len(values), self.constant, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_bundle_files(tmp_path: Path) -> tuple[Path, Path]:
    contract = FeatureContract.load(CONTRACT_PATH)
    model = FakeProbabilityModel(len(contract.model_features))
    model_path = tmp_path / "fake.joblib"
    joblib.dump(model, model_path)
    manifest = deepcopy(json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8")))
    manifest["artifact"] = {
        "filename": model_path.name,
        "sha256": sha256(model_path),
        "size_bytes": model_path.stat().st_size,
        "distribution": "test",
    }
    manifest["serialization"] = {
        "python_version": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "required_package_versions": {
            package: metadata.version(package)
            for package in ("scikit-learn", "numpy", "joblib")
        },
        "runtime_requirements_file": "requirements-runtime.txt",
    }
    manifest["model"] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "classes": [0, 1],
        "score_semantics": "test",
        "frozen_parameters": {"constant": 0.25},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path, manifest_path


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_loader_verifies_bundle_and_predicts_attack_scores(tmp_path: Path) -> None:
    model_path, manifest_path = fake_bundle_files(tmp_path)
    bundle = FrozenModelBundle.load(model_path, manifest_path, CONTRACT_PATH)
    scores = bundle.predict_scores(np.zeros((3, 128), dtype=np.float32))

    np.testing.assert_allclose(scores, [0.25, 0.25, 0.25])
    assert bundle.metadata()["feature_contract"] == "cicflow-v2-128"


def test_hash_mismatch_fails_before_joblib_deserialization(tmp_path: Path) -> None:
    model_path, manifest_path = fake_bundle_files(tmp_path)
    model_path.write_bytes(model_path.read_bytes() + b"corruption")

    with pytest.raises(ArtifactIntegrityError, match="SHA256 mismatch"):
        FrozenModelBundle.load(model_path, manifest_path, CONTRACT_PATH)


def test_feature_contract_version_mismatch_fails_closed(tmp_path: Path) -> None:
    model_path, manifest_path = fake_bundle_files(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatibility"]["feature_contract"] = "wrong-contract"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="version mismatch"):
        FrozenModelBundle.load(model_path, manifest_path, CONTRACT_PATH)


def test_serialization_runtime_mismatch_fails_closed(tmp_path: Path) -> None:
    model_path, manifest_path = fake_bundle_files(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["serialization"]["required_package_versions"]["scikit-learn"] = "0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="runtime mismatch"):
        FrozenModelBundle.load(model_path, manifest_path, CONTRACT_PATH)


@pytest.mark.skipif(
    not FROZEN_MODEL_PATH.is_file(),
    reason="Frozen binary is distributed as a GitHub Release asset, not in Git history",
)
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_local_frozen_release_artifact_loads_with_exact_metadata() -> None:
    bundle = FrozenModelBundle.load(
        FROZEN_MODEL_PATH, FROZEN_MANIFEST_PATH, CONTRACT_PATH
    )
    assert bundle.model_version == "context-rf-v2"
    assert bundle.manifest["artifact"]["sha256"] == (
        "4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72"
    )
    assert len(bundle.contract.model_features) == 128
