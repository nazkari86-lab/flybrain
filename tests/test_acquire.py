import hashlib
import io
from pathlib import Path

import pytest

from flybrain.acquire import (
    TEN_GIB,
    ChecksumMismatch,
    DiskBudgetError,
    acquire_artifact,
    artifact_path,
)
from flybrain.manifest import Artifact


def artifact_for(content: bytes, *, declared: bytes | None = None) -> Artifact:
    checksum_content = content if declared is None else declared
    return Artifact(
        url="https://example.org/data.bin",
        bytes=len(content),
        sha256=hashlib.sha256(checksum_content).hexdigest(),
    )


def test_acquire_refuses_when_ten_gib_reserve_would_be_crossed(tmp_path: Path) -> None:
    artifact = artifact_for(b"abc")

    with pytest.raises(DiskBudgetError):
        acquire_artifact(
            artifact,
            tmp_path,
            free_bytes=lambda _: TEN_GIB + 2,
            opener=lambda _: io.BytesIO(b"abc"),
        )


def test_acquire_promotes_verified_content_atomically(tmp_path: Path) -> None:
    artifact = artifact_for(b"connectome")

    result = acquire_artifact(
        artifact,
        tmp_path,
        free_bytes=lambda _: TEN_GIB + 100,
        opener=lambda _: io.BytesIO(b"connectome"),
    )

    assert result == artifact_path(tmp_path, artifact)
    assert result.read_bytes() == b"connectome"
    assert not list(tmp_path.rglob("*.partial"))


def test_acquire_quarantines_checksum_mismatch(tmp_path: Path) -> None:
    artifact = artifact_for(b"bad", declared=b"good")

    with pytest.raises(ChecksumMismatch):
        acquire_artifact(
            artifact,
            tmp_path,
            free_bytes=lambda _: TEN_GIB + 100,
            opener=lambda _: io.BytesIO(b"bad"),
        )

    assert not artifact_path(tmp_path, artifact).exists()
    assert len(list((tmp_path / "quarantine").glob("*.partial"))) == 1


def test_acquire_uses_verified_cache_without_opening_source(tmp_path: Path) -> None:
    artifact = artifact_for(b"cached")
    target = artifact_path(tmp_path, artifact)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached")

    result = acquire_artifact(
        artifact,
        tmp_path,
        free_bytes=lambda _: 0,
        opener=lambda _: (_ for _ in ()).throw(AssertionError("source opened")),
    )

    assert result == target
