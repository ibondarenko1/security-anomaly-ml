from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from pathlib import Path

from src.security_anomaly.package_resources import (
    FEATURE_CONTRACT_RESOURCE,
    INCIDENT_SCHEMA_RESOURCE,
    MODEL_MANIFEST_RESOURCE,
    resource_ref,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_only_product_src_package_and_console_script() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "0.1.0"
    assert document["project"]["requires-python"] == ">=3.13,<3.14"
    assert document["project"]["scripts"]["security-anomaly"] == (
        "security_anomaly.cli:main"
    )
    assert document["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "security_anomaly",
        "security_anomaly.*",
    ]


def test_all_installed_runtime_resources_resolve() -> None:
    for name in (
        FEATURE_CONTRACT_RESOURCE,
        MODEL_MANIFEST_RESOURCE,
        INCIDENT_SCHEMA_RESOURCE,
    ):
        resource = resource_ref(name)
        assert resource.is_file()
        assert resource.read_bytes()


def test_checkout_feature_contract_bytes_match_frozen_manifest() -> None:
    manifest = json.loads(
        (ROOT / "contracts" / "model-manifest-context-rf-v2.json").read_text(
            encoding="utf-8"
        )
    )
    expected = manifest["compatibility"]["feature_contract_sha256"]
    repository_bytes = (
        ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
    ).read_bytes()
    package_bytes = (
        ROOT
        / "src"
        / "security_anomaly"
        / "resources"
        / "feature-contract-cicflow-v2-128.json"
    ).read_bytes()
    assert hashlib.sha256(repository_bytes).hexdigest() == expected
    assert hashlib.sha256(package_bytes).hexdigest() == expected


def test_runtime_package_has_no_research_training_or_holdout_imports() -> None:
    forbidden = (
        "train_",
        "evaluate_",
        "select_threshold",
        "build_temporal_features",
        "attack_cat",
        "feb18",
    )
    for path in (ROOT / "src" / "security_anomaly").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert not any(value in source.lower() for value in forbidden), path
