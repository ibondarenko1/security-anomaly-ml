"""Portable access to immutable runtime contracts bundled in the wheel."""

from __future__ import annotations

import json
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterator


FEATURE_CONTRACT_RESOURCE = "feature-contract-cicflow-v2-128.json"
MODEL_MANIFEST_RESOURCE = "model-manifest-context-rf-v2.json"
INCIDENT_SCHEMA_RESOURCE = "incident-v1.schema.json"


def resource_ref(name: str):
    """Return an importlib Traversable for a bundled, versioned resource."""

    return files(f"{__package__}.resources").joinpath(name)


def resource_bytes(name: str) -> bytes:
    return resource_ref(name).read_bytes()


def resource_json(name: str) -> dict[str, Any]:
    value = json.loads(resource_ref(name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Package resource must contain a JSON object: {name}")
    return value


@contextmanager
def materialized_resource(name: str) -> Iterator[Path]:
    """Expose a resource as a real path for existing integrity-checked loaders."""

    with as_file(resource_ref(name)) as path:
        yield Path(path)
