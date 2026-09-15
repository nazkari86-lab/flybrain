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
