"""Guarded acquisition of immutable, content-addressed source artifacts."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast
from urllib.request import urlopen

from flybrain.manifest import Artifact

TEN_GIB = 10 * 1024**3


class AcquisitionError(RuntimeError):
    """Base class for source acquisition failures."""


class DiskBudgetError(AcquisitionError):
    """The artifact would cross the mandatory free-space reserve."""


class ChecksumMismatch(AcquisitionError):
    """Retrieved bytes do not match the declared digest."""


def artifact_path(root: Path, artifact: Artifact) -> Path:
    """Return the stable content-addressed path for an artifact."""

    return root / "sha256" / artifact.sha256[:2] / artifact.sha256


def _available_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _open_url(url: str) -> BinaryIO:
    return cast(BinaryIO, urlopen(url))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_artifact(
    artifact: Artifact,
    root: Path,
    *,
    free_bytes: Callable[[Path], int] = _available_bytes,
    opener: Callable[[str], BinaryIO] = _open_url,
) -> Path:
    """Retrieve, verify, and atomically promote an artifact into the cache."""

    root.mkdir(parents=True, exist_ok=True)
    target = artifact_path(root, artifact)
    if target.is_file() and _sha256(target) == artifact.sha256:
        return target

    if free_bytes(root) - artifact.bytes < TEN_GIB:
        raise DiskBudgetError(
            f"artifact requires {artifact.bytes} bytes while preserving a 10 GiB reserve"
        )

    quarantine = root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    partial = quarantine / f"{artifact.sha256}.partial"
    digest = hashlib.sha256()

    with opener(str(artifact.url)) as source, partial.open("wb") as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(chunk)
            digest.update(chunk)

    if digest.hexdigest() != artifact.sha256:
        raise ChecksumMismatch(f"checksum mismatch for {artifact.url}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(target)
    return target
