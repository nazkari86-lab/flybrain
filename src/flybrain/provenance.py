"""Immutable identity validation for canonical connectome snapshots."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from flybrain.plasticity import PlasticStateIdentity

SNAPSHOT_IDENTITY_FILES = (
    "metadata.json",
    "source-annotations.parquet",
    "edges.parquet",
)
SNAPSHOT_CONTENT_FILES = (
    "metadata.json",
    "neurons.parquet",
    "edges.parquet",
    "source-annotations.parquet",
)


class _SnapshotIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dataset_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def snapshot_sha256(snapshot: Path) -> str:
    """Hash the identity-bearing snapshot files in their canonical byte order."""

    digest = hashlib.sha256()
    for name in SNAPSHOT_IDENTITY_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        with (snapshot / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def snapshot_content_sha256(snapshot: Path) -> str:
    """Hash every available canonical snapshot component in stable name order."""

    available = {name for name in SNAPSHOT_CONTENT_FILES if (snapshot / name).is_file()}
    required = {"metadata.json", "neurons.parquet", "edges.parquet"}
    if not required.issubset(available):
        raise ValueError("snapshot requires metadata, neurons, and edges for content identity")
    digest = hashlib.sha256()
    for name in SNAPSHOT_CONTENT_FILES:
        if name not in available:
            continue
        path = snapshot / name
        digest.update(name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_state_identity(snapshot: Path, identity: PlasticStateIdentity) -> None:
    """Reject a plastic state whose immutable source differs from the snapshot."""

    metadata = _SnapshotIdentity.model_validate_json(
        (snapshot / "metadata.json").read_text(encoding="utf-8")
    )
    mismatches = []
    if metadata.dataset_id != identity.dataset_id:
        mismatches.append("dataset_id")
    if metadata.manifest_sha256 != identity.source_manifest_sha256:
        mismatches.append("source_manifest_sha256")
    if snapshot_sha256(snapshot) != identity.snapshot_sha256:
        mismatches.append("snapshot_sha256")
    if mismatches:
        raise ValueError(f"snapshot identity mismatch: {', '.join(mismatches)}")
