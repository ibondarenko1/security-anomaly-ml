from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.validate_release import (
    ReleaseValidationError,
    validate_release,
    version_from_tag,
)


def _write_dist(root: Path, version: str = "1.2.3") -> Path:
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / f"security_anomaly_ml-{version}-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: security-anomaly-ml\n"
        f"Version: {version}\n\n"
    ).encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"security_anomaly_ml-{version}.dist-info/METADATA", metadata
        )
    sdist = dist / f"security_anomaly_ml-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = io.BytesIO(metadata)
        info = tarfile.TarInfo(f"security_anomaly_ml-{version}/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, payload)
        nested = io.BytesIO(metadata)
        nested_info = tarfile.TarInfo(
            f"security_anomaly_ml-{version}/src/security_anomaly_ml.egg-info/PKG-INFO"
        )
        nested_info.size = len(metadata)
        archive.addfile(nested_info, nested)
    return dist


def _write_project(root: Path, version: str = "1.2.3") -> None:
    (root / "pyproject.toml").write_text(
        f"[project]\nname = 'security-anomaly-ml'\nversion = '{version}'\n",
        encoding="utf-8",
    )


def test_stable_tag_is_strict() -> None:
    assert version_from_tag("v1.2.3") == "1.2.3"
    for tag in ("1.2.3", "v1.2.3.dev1", "v1.2", "v1.2.3+local"):
        with pytest.raises(ReleaseValidationError):
            version_from_tag(tag)


def test_release_binds_project_and_archive_metadata(tmp_path: Path) -> None:
    _write_project(tmp_path)
    dist = _write_dist(tmp_path)
    result = validate_release(tmp_path, "v1.2.3", dist)
    assert result["version"] == "1.2.3"
    assert set(result["sha256"]) == {
        "security_anomaly_ml-1.2.3-py3-none-any.whl",
        "security_anomaly_ml-1.2.3.tar.gz",
    }
    assert (dist / "SHA256SUMS").read_text(encoding="utf-8").count("\n") == 2


def test_release_rejects_project_mismatch(tmp_path: Path) -> None:
    _write_project(tmp_path, "1.2.4")
    with pytest.raises(ReleaseValidationError, match="does not match"):
        validate_release(tmp_path, "v1.2.3")


def test_release_rejects_archive_metadata_mismatch(tmp_path: Path) -> None:
    _write_project(tmp_path)
    dist = _write_dist(tmp_path, "1.2.3")
    wheel = dist / "security_anomaly_ml-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "security_anomaly_ml-1.2.3.dist-info/METADATA",
            b"Metadata-Version: 2.4\nName: security-anomaly-ml\nVersion: 1.2.4\n",
        )
    with pytest.raises(ReleaseValidationError, match="metadata version"):
        validate_release(tmp_path, "v1.2.3", dist)
