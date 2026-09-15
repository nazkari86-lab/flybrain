"""Persistent dopamine-gated eligibility plasticity for selected synapses."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlasticityParameters:
    """Explicit fitted assumptions for bounded three-factor plasticity."""

    eligibility_tau_ms: float = 1000.0
    learning_rate: float = 0.05
    min_multiplier: float = 0.2
    max_multiplier: float = 1.5

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("plasticity parameters must be finite")
        if self.eligibility_tau_ms <= 0:
            raise ValueError("eligibility_tau_ms must be positive")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.min_multiplier <= 1 <= self.max_multiplier:
            raise ValueError("multiplier bounds must contain the baseline value 1")


@dataclass(frozen=True)
class PlasticStateIdentity:
    """Immutable snapshot identity bound to a persisted plastic overlay."""

    dataset_id: str
    source_manifest_sha256: str
    snapshot_sha256: str
    format_version: int = 1

    def validate(self) -> None:
        if not self.dataset_id:
            raise ValueError("plastic state dataset_id must not be empty")
        if self.format_version != 1:
            raise ValueError(f"unsupported plastic state format {self.format_version}")
        for name in ("source_manifest_sha256", "snapshot_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"plastic state {name} must be a SHA-256 digest")


@dataclass
class PlasticEdgeSet:
    """Fixed anatomical edges plus mutable eligibility and weight multipliers."""

    pre_ids: NDArray[np.uint64]
    post_ids: NDArray[np.uint64]
    baseline_weights: NDArray[np.float32]
    multipliers: NDArray[np.float32]
    eligibility: NDArray[np.float32]

    @classmethod
    def create(
        cls,
        *,
        pre_ids: NDArray[np.uint64],
        post_ids: NDArray[np.uint64],
        baseline_weights: NDArray[np.float32],
    ) -> PlasticEdgeSet:
        if pre_ids.ndim != 1 or post_ids.ndim != 1 or baseline_weights.ndim != 1:
            raise ValueError("plastic edge arrays must be one-dimensional")
        if not (pre_ids.size == post_ids.size == baseline_weights.size):
            raise ValueError("plastic edge arrays must have equal length")
        if not np.all(np.isfinite(baseline_weights)):
            raise ValueError("plastic baseline weights must be finite")
        if np.any(baseline_weights <= 0):
            raise ValueError("plastic baseline weights must be positive")
        return cls(
            pre_ids=pre_ids.astype(np.uint64, copy=True),
            post_ids=post_ids.astype(np.uint64, copy=True),
            baseline_weights=baseline_weights.astype(np.float32, copy=True),
            multipliers=np.ones(pre_ids.size, dtype=np.float32),
            eligibility=np.zeros(pre_ids.size, dtype=np.float32),
        )

    @property
    def effective_weights(self) -> NDArray[np.float32]:
        return self.baseline_weights * self.multipliers

    def update_eligibility(
        self,
        *,
        active_pre_ids: NDArray[np.uint64],
        gated_post_ids: NDArray[np.uint64],
        dt_ms: float,
        params: PlasticityParameters,
    ) -> None:
        """Decay existing eligibility and mark coincident pre/post-gated edges."""

        params.validate()
        if dt_ms < 0:
            raise ValueError("dt_ms must be non-negative")
        decay = np.float32(np.exp(-dt_ms / params.eligibility_tau_ms))
        self.eligibility *= decay
        if active_pre_ids.size and gated_post_ids.size:
            eligible = np.isin(self.pre_ids, active_pre_ids) & np.isin(
                self.post_ids, gated_post_ids
            )
            self.eligibility[eligible] += 1.0

    def clear_eligibility(self) -> None:
        self.eligibility.fill(0.0)

    def apply_dopamine(
        self,
        dopamine_by_post: dict[int, float],
        params: PlasticityParameters,
    ) -> None:
        """Apply signed dopamine only where a recent eligibility trace exists."""

        params.validate()
        if not dopamine_by_post:
            return
        if not all(math.isfinite(value) for value in dopamine_by_post.values()):
            raise ValueError("dopamine values must be finite")
        modulation = np.fromiter(
            (dopamine_by_post.get(int(post_id), 0.0) for post_id in self.post_ids),
            dtype=np.float32,
            count=self.post_ids.size,
        )
        self.multipliers -= np.float32(params.learning_rate) * modulation * self.eligibility
        np.clip(
            self.multipliers,
            params.min_multiplier,
            params.max_multiplier,
            out=self.multipliers,
        )


def save_plastic_state(
    path: Path,
    edges: PlasticEdgeSet,
    params: PlasticityParameters,
    identity: PlasticStateIdentity,
) -> None:
    """Atomically persist anatomy identity, mutable memory, and rule parameters."""

    params.validate()
    identity.validate()
    _validate_edge_state(edges, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(
            stream,
            pre_ids=edges.pre_ids,
            post_ids=edges.post_ids,
            baseline_weights=edges.baseline_weights,
            multipliers=edges.multipliers,
            eligibility=edges.eligibility,
            parameters=np.array(json.dumps(asdict(params), sort_keys=True)),
            identity=np.array(json.dumps(asdict(identity), sort_keys=True)),
        )
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def load_plastic_state(
    path: Path,
) -> tuple[PlasticEdgeSet, PlasticityParameters, PlasticStateIdentity]:
    """Restore a plastic edge overlay without pickle or implicit defaults."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            params = PlasticityParameters(**json.loads(str(archive["parameters"].item())))
            params.validate()
            identity = PlasticStateIdentity(**json.loads(str(archive["identity"].item())))
            identity.validate()
            arrays = [
                archive["pre_ids"],
                archive["post_ids"],
                archive["baseline_weights"],
                archive["multipliers"],
                archive["eligibility"],
            ]
            if any(array.ndim != 1 for array in arrays):
                raise ValueError("plastic state arrays must be one-dimensional")
            if len({array.shape for array in arrays}) != 1:
                raise ValueError("plastic state arrays have inconsistent shapes")
            edges = PlasticEdgeSet(
                pre_ids=arrays[0].astype(np.uint64, copy=True),
                post_ids=arrays[1].astype(np.uint64, copy=True),
                baseline_weights=arrays[2].astype(np.float32, copy=True),
                multipliers=arrays[3].astype(np.float32, copy=True),
                eligibility=arrays[4].astype(np.float32, copy=True),
            )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"cannot load plastic state {path}: {error}") from error
    _validate_edge_state(edges, params)
    return edges, params, identity


def _validate_edge_state(edges: PlasticEdgeSet, params: PlasticityParameters) -> None:
    arrays = (
        edges.pre_ids,
        edges.post_ids,
        edges.baseline_weights,
        edges.multipliers,
        edges.eligibility,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("plastic state arrays must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("plastic state arrays have inconsistent shapes")
    if not np.all(np.isfinite(edges.baseline_weights)) or np.any(edges.baseline_weights <= 0):
        raise ValueError("plastic state baseline weights must be finite and positive")
    if not np.all(np.isfinite(edges.multipliers)):
        raise ValueError("plastic state multipliers must be finite")
    if not np.all(np.isfinite(edges.eligibility)) or np.any(edges.eligibility < 0):
        raise ValueError("plastic state eligibility must be finite and non-negative")
    if np.any(edges.multipliers < params.min_multiplier) or np.any(
        edges.multipliers > params.max_multiplier
    ):
        raise ValueError("plastic state multipliers violate configured bounds")
