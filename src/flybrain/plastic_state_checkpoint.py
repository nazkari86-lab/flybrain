"""Versioned, identity-bound persistence for sparse plastic overlays."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.provenance import snapshot_content_sha256


class PlasticStateCheckpoint(BaseModel, frozen=True):
    """Auditable sparse plastic state bound to exact immutable inputs."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["plastic-state-checkpoint-v1"] = "plastic-state-checkpoint-v1"
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    edge_indices: tuple[int, ...]
    multipliers: tuple[float, ...]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _payload(
    overlay: PlasticWeightOverlay,
    snapshot_hash: str,
    registry_sha256: str,
) -> dict[str, object]:
    return {
        "protocol": "plastic-state-checkpoint-v1",
        "snapshot_content_sha256": snapshot_hash,
        "registry_sha256": registry_sha256,
        "edge_indices": tuple(int(value) for value in overlay.edge_indices),
        "multipliers": tuple(float(value) for value in overlay.multipliers),
    }


def save_plastic_state(
    path: Path,
    overlay: PlasticWeightOverlay,
    snapshot: Path,
    registry_sha256: str,
) -> PlasticStateCheckpoint:
    """Atomically create one new checkpoint and refuse overwrite."""

    if path.exists():
        raise FileExistsError(f"plastic checkpoint already exists: {path}")
    if len(registry_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in registry_sha256
    ):
        raise ValueError("registry_sha256 must be a lowercase SHA-256 digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_hash = snapshot_content_sha256(snapshot)
    payload = _payload(overlay, snapshot_hash, registry_sha256)
    checkpoint = PlasticStateCheckpoint(
        snapshot_content_sha256=snapshot_hash,
        registry_sha256=registry_sha256,
        edge_indices=tuple(int(value) for value in overlay.edge_indices),
        multipliers=tuple(float(value) for value in overlay.multipliers),
        digest=_digest(payload),
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(json.dumps(checkpoint.model_dump(mode="json"), sort_keys=True))
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if path.exists():
            raise FileExistsError(f"plastic checkpoint already exists: {path}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint


def load_plastic_state(
    path: Path,
    binding: PlasticEdgeBinding,
    snapshot: Path,
    registry_sha256: str,
) -> PlasticWeightOverlay:
    """Load only a checkpoint matching current snapshot, registry, shape, and digest."""

    checkpoint = PlasticStateCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.snapshot_content_sha256 != snapshot_content_sha256(snapshot):
        raise ValueError("plastic checkpoint snapshot identity mismatch")
    if checkpoint.registry_sha256 != registry_sha256:
        raise ValueError("plastic checkpoint registry identity mismatch")
    payload = checkpoint.model_dump(mode="json", exclude={"digest"})
    if _digest(payload) != checkpoint.digest:
        raise ValueError("plastic checkpoint digest mismatch")
    if tuple(checkpoint.edge_indices) != tuple(
        int(value) for value in binding.overlay.edge_indices
    ):
        raise ValueError("plastic checkpoint edge locations mismatch")
    if len(checkpoint.multipliers) != binding.overlay.multipliers.size:
        raise ValueError("plastic checkpoint multiplier shape mismatch")
    restored = binding.overlay.copy()
    restored.set_multipliers(
        np.asarray(checkpoint.multipliers, dtype=np.float32), minimum=0.0, maximum=2.0
    )
    return restored
