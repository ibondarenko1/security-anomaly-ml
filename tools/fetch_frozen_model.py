"""Atomically fetch and verify the exact public frozen model release asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO


DEFAULT_REPOSITORY = "ibondarenko1/security-anomaly-ml"
MODEL_FILENAME = "context-rf-v2.joblib"
FROZEN_MODEL_SHA256 = (
    "4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72"
)
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ModelFetchError(RuntimeError):
    """Raised when public model acquisition cannot fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_asset_url(repository: str, tag: str) -> str:
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise ModelFetchError("Repository must use the exact owner/name form")
    if not tag or tag.strip() != tag:
        raise ModelFetchError("Release tag must be a non-empty exact value")
    encoded_tag = urllib.parse.quote(tag, safe="")
    return (
        f"https://github.com/{repository}/releases/download/"
        f"{encoded_tag}/{MODEL_FILENAME}"
    )


def _copy_response(response: BinaryIO, destination: BinaryIO) -> None:
    while True:
        block = response.read(1024 * 1024)
        if not block:
            break
        destination.write(block)


def fetch_model(
    *,
    repository: str,
    tag: str,
    destination: Path,
    expected_sha256: str = FROZEN_MODEL_SHA256,
) -> dict[str, object]:
    """Fetch once, verify before success, and never replace an existing file."""

    expected_sha256 = expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ModelFetchError("Expected SHA256 must be 64 lowercase hexadecimal characters")
    destination = destination.expanduser().resolve()
    if destination.exists():
        if not destination.is_file():
            raise ModelFetchError(f"Destination exists and is not a file: {destination}")
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ModelFetchError(
                "Refusing to replace existing model with unexpected SHA256: "
                f"expected {expected_sha256}, got {actual}"
            )
        return {
            "path": str(destination),
            "sha256": actual,
            "downloaded": False,
            "verified": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = release_asset_url(repository, tag)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
        )
        temporary = Path(temporary_name)
        request = urllib.request.Request(
            url, headers={"User-Agent": "security-anomaly-ml-model-fetch/0.1.0"}
        )
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=120) as response:
                _copy_response(response, output)
            output.flush()
            os.fsync(output.fileno())
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ModelFetchError(
                f"Downloaded model SHA256 mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
        temporary = None
        return {
            "path": str(destination),
            "sha256": actual,
            "downloaded": True,
            "verified": True,
            "release_tag": tag,
            "url": url,
        }
    except (OSError, urllib.error.URLError) as error:
        raise ModelFetchError(f"Unable to download frozen model from {url}: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--tag", required=True, help="exact dedicated model release tag")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--sha256", default=FROZEN_MODEL_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = fetch_model(
            repository=args.repository,
            tag=args.tag,
            destination=args.destination,
            expected_sha256=args.sha256,
        )
    except ModelFetchError as error:
        raise SystemExit(f"model fetch failed: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
