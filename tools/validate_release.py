"""Validate a stable release tag and the distributions built from it.

The release workflows call this module before a package or container can be
published.  It keeps the version binding in one place and inspects metadata
inside both archives, so a filename or shell variable cannot make a different
project version look publishable.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path


STABLE_TAG = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")
PROJECT_NAME = "security-anomaly-ml"
NORMALIZED_PROJECT_NAME = PROJECT_NAME.replace("-", "_")


class ReleaseValidationError(ValueError):
    """Raised when a release input does not satisfy the release contract."""


def read_project_version(project_root: Path) -> str:
    """Read the PEP 621 project version from ``pyproject.toml``."""

    try:
        document = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseValidationError(f"cannot read pyproject.toml: {error}") from error
    project = document.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version:
        raise ReleaseValidationError("pyproject.toml has no static project version")
    return version


def version_from_tag(tag: str) -> str:
    """Return a stable semantic version from a release tag."""

    match = STABLE_TAG.fullmatch(tag)
    if match is None:
        raise ReleaseValidationError(
            f"release tag must be a stable vMAJOR.MINOR.PATCH tag, got {tag!r}"
        )
    return match.group("version")


def _metadata_version(text: str, source: Path) -> str:
    for line in text.splitlines():
        if line.startswith("Version:"):
            version = line.partition(":")[2].strip()
            if version:
                return version
    raise ReleaseValidationError(f"{source.name} has no Version metadata")


def _metadata_name(text: str, source: Path) -> str:
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line.partition(":")[2].strip()
            if name:
                return name
    raise ReleaseValidationError(f"{source.name} has no Name metadata")


def _wheel_metadata(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ReleaseValidationError(
                    f"{path.name} must contain exactly one dist-info/METADATA"
                )
            return archive.read(metadata_names[0]).decode("utf-8")
    except (OSError, UnicodeError, zipfile.BadZipFile) as error:
        if isinstance(error, ReleaseValidationError):
            raise
        raise ReleaseValidationError(f"cannot inspect {path.name}: {error}") from error


def _sdist_metadata(path: Path) -> str:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            metadata_names = [
                member
                for member in archive.getmembers()
                if len(member.name.rstrip("/").split("/")) == 2
                and member.name.rstrip("/").split("/")[-1] == "PKG-INFO"
            ]
            if len(metadata_names) != 1:
                raise ReleaseValidationError(
                    f"{path.name} must contain exactly one top-level PKG-INFO"
                )
            handle = archive.extractfile(metadata_names[0])
            if handle is None:
                raise ReleaseValidationError(f"cannot read PKG-INFO from {path.name}")
            return handle.read().decode("utf-8")
    except (OSError, UnicodeError, tarfile.TarError) as error:
        if isinstance(error, ReleaseValidationError):
            raise
        raise ReleaseValidationError(f"cannot inspect {path.name}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_distributions(dist_dir: Path, version: str) -> dict[str, str]:
    """Validate one wheel and one source distribution and return their hashes."""

    if not dist_dir.is_dir():
        raise ReleaseValidationError(f"distribution directory not found: {dist_dir}")
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseValidationError(
            "dist must contain exactly one wheel and one .tar.gz source distribution"
        )
    wheel, sdist = wheels[0], sdists[0]
    expected_prefix = f"{NORMALIZED_PROJECT_NAME}-{version}"
    if not wheel.name.startswith(expected_prefix + "-"):
        raise ReleaseValidationError(
            f"wheel filename {wheel.name!r} does not match {expected_prefix!r}"
        )
    if not sdist.name.startswith(expected_prefix + ".tar.gz"):
        raise ReleaseValidationError(
            f"source filename {sdist.name!r} does not match {expected_prefix!r}"
        )
    for archive, metadata_reader in ((wheel, _wheel_metadata), (sdist, _sdist_metadata)):
        metadata = metadata_reader(archive)
        metadata_name = _metadata_name(metadata, archive)
        if metadata_name != PROJECT_NAME:
            raise ReleaseValidationError(
                f"{archive.name} metadata name {metadata_name!r} != {PROJECT_NAME!r}"
            )
        metadata_version = _metadata_version(metadata, archive)
        if metadata_version != version:
            raise ReleaseValidationError(
                f"{archive.name} metadata version {metadata_version!r} != {version!r}"
            )
    return {path.name: _sha256(path) for path in (wheel, sdist)}


def write_sha256sums(dist_dir: Path, hashes: dict[str, str]) -> Path:
    """Write a deterministic GNU-compatible checksum manifest."""

    destination = dist_dir / "SHA256SUMS"
    destination.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )
    return destination


def validate_release(project_root: Path, tag: str, dist_dir: Path | None = None) -> dict[str, object]:
    """Validate tag/project agreement and, when supplied, distribution archives."""

    version = version_from_tag(tag)
    project_version = read_project_version(project_root)
    if project_version != version:
        raise ReleaseValidationError(
            f"project version {project_version!r} does not match tag {tag!r}"
        )
    result: dict[str, object] = {"tag": tag, "version": version}
    if dist_dir is not None:
        hashes = validate_distributions(dist_dir, version)
        result["sha256"] = hashes
        result["sha256sums"] = str(write_sha256sums(dist_dir, hashes))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = validate_release(
            args.project_root.expanduser().resolve(),
            args.tag,
            args.dist_dir.expanduser().resolve() if args.dist_dir else None,
        )
    except ReleaseValidationError as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 2
    import json

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
