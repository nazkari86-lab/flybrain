"""Atomic and configuration-bound LIF simulation checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from flybrain.dynamics import LIFState


class CheckpointError(RuntimeError):
    """A checkpoint cannot be trusted or restored."""


class ConfigHashMismatch(CheckpointError):
    """A checkpoint belongs to a different experiment configuration."""


class CheckpointMetadata(BaseModel):
    """Identity required to validate continuation of a run."""

    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1)
    seed: int


def save_checkpoint(path: Path, state: LIFState, metadata: CheckpointMetadata) -> None:
    """Atomically save all mutable dynamics state and its identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(
            stream,
            voltage=state.voltage,
            recurrent_current=state.recurrent_current,
            step=np.array(state.step, dtype=np.int64),
            rng_state=np.array(json.dumps(state.rng_state)),
            metadata=np.array(metadata.model_dump_json()),
        )
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def load_checkpoint(
    path: Path,
    expected_config_hash: str,
) -> tuple[LIFState, CheckpointMetadata]:
    """Load a checkpoint only when its structure and configuration match."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = CheckpointMetadata.model_validate_json(str(archive["metadata"].item()))
            state = LIFState(
                voltage=archive["voltage"].astype(np.float32, copy=True),
                recurrent_current=archive["recurrent_current"].astype(np.float32, copy=True),
                step=int(archive["step"].item()),
                rng_state=json.loads(str(archive["rng_state"].item())),
            )
    except (OSError, ValueError, KeyError) as error:
        raise CheckpointError(f"cannot load checkpoint {path}") from error

    if metadata.config_hash != expected_config_hash:
        raise ConfigHashMismatch(
            f"checkpoint config {metadata.config_hash} does not match {expected_config_hash}"
        )
    return state, metadata
