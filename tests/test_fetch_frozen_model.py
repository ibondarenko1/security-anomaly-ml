from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tools import fetch_frozen_model


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_fetch_is_atomic_verified_and_uses_explicit_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic test model bytes"
    observed: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return Response(payload)

    monkeypatch.setattr(fetch_frozen_model.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "models" / fetch_frozen_model.MODEL_FILENAME
    result = fetch_frozen_model.fetch_model(
        repository="owner/repository",
        tag="model-explicit-tag",
        destination=destination,
        expected_sha256=digest(payload),
    )
    assert destination.read_bytes() == payload
    assert result["downloaded"] is True
    assert result["verified"] is True
    assert observed["url"] == (
        "https://github.com/owner/repository/releases/download/"
        "model-explicit-tag/context-rf-v2.joblib"
    )
    assert not list(destination.parent.glob("*.download"))


def test_hash_mismatch_leaves_no_destination_or_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fetch_frozen_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b"wrong bytes"),
    )
    destination = tmp_path / fetch_frozen_model.MODEL_FILENAME
    with pytest.raises(fetch_frozen_model.ModelFetchError, match="SHA256 mismatch"):
        fetch_frozen_model.fetch_model(
            repository="owner/repository",
            tag="model-tag",
            destination=destination,
            expected_sha256="0" * 64,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.download"))


def test_existing_verified_file_is_not_downloaded_or_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"already verified"
    destination = tmp_path / fetch_frozen_model.MODEL_FILENAME
    destination.write_bytes(payload)
    monkeypatch.setattr(
        fetch_frozen_model.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("download attempted"),
    )
    result = fetch_frozen_model.fetch_model(
        repository="owner/repository",
        tag="model-tag",
        destination=destination,
        expected_sha256=digest(payload),
    )
    assert result["downloaded"] is False
    assert destination.read_bytes() == payload


def test_existing_wrong_file_fails_without_replacement(tmp_path: Path) -> None:
    destination = tmp_path / fetch_frozen_model.MODEL_FILENAME
    destination.write_bytes(b"do not overwrite")
    with pytest.raises(fetch_frozen_model.ModelFetchError, match="Refusing to replace"):
        fetch_frozen_model.fetch_model(
            repository="owner/repository",
            tag="model-tag",
            destination=destination,
            expected_sha256="0" * 64,
        )
    assert destination.read_bytes() == b"do not overwrite"
