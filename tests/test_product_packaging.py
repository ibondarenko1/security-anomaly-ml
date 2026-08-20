from __future__ import annotations

import tomllib
import ast
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
