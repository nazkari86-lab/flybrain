"""Sparse, sign-preserving multipliers for predeclared canonical graph edges."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlasticWeightOverlay:
    """Mutable learning state kept separate from immutable canonical edge weights."""

    edge_indices: NDArray[np.int64]
    multipliers: NDArray[np.float32]
    canonical_edge_count: int

    @classmethod
    def create(
        cls,
        *,
        edge_indices: NDArray[np.int64],
        canonical_edge_count: int,
    ) -> PlasticWeightOverlay:
        if canonical_edge_count <= 0:
            raise ValueError("canonical edge count must be positive")
        indices = edge_indices.astype(np.int64, copy=True)
        if indices.ndim != 1 or not indices.size:
            raise ValueError("plastic edge indices must be nonempty and one-dimensional")
        if int(indices.min()) < 0 or int(indices.max()) >= canonical_edge_count:
            raise ValueError("plastic edge index outside canonical graph")
        if not np.all(indices[:-1] < indices[1:]):
            raise ValueError("plastic edge indices must be unique and sorted")
        return cls(
            edge_indices=indices,
            multipliers=np.ones(indices.size, dtype=np.float32),
            canonical_edge_count=canonical_edge_count,
        )

    def apply_to(self, canonical_weights: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return effective weights without modifying the canonical graph data array."""

        if canonical_weights.shape != (self.canonical_edge_count,):
            raise ValueError("canonical weights do not match graph edge count")
        effective = canonical_weights.astype(np.float32, copy=True)
        effective[self.edge_indices] *= self.multipliers
        return effective

    def set_multipliers(
        self,
        multipliers: NDArray[np.float32],
        *,
        minimum: float,
        maximum: float,
    ) -> None:
        """Replace only this overlay's bounded, finite edge multipliers."""

        values = multipliers.astype(np.float32, copy=False)
        if values.shape != self.multipliers.shape:
            raise ValueError("plastic multiplier shape differs from edge manifest")
        if not np.all(np.isfinite(values)):
            raise ValueError("plastic multipliers must be finite")
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("plastic multiplier bounds must be finite")
        if minimum > 1.0 or maximum < 1.0 or minimum > maximum:
            raise ValueError("plastic multiplier bounds must contain one")
        if np.any(values < minimum) or np.any(values > maximum):
            raise ValueError("plastic multipliers violate configured bounds")
        self.multipliers[:] = values

    def reset(self) -> None:
        """Restore the exact unlearned multiplier state."""

        self.multipliers.fill(1.0)

    def copy(self) -> PlasticWeightOverlay:
        """Clone sparse mutable state without aliasing either array."""

        return PlasticWeightOverlay(
            edge_indices=self.edge_indices.copy(),
            multipliers=self.multipliers.copy(),
            canonical_edge_count=self.canonical_edge_count,
        )

    def digest(self) -> str:
        """Return a deterministic identity of sparse locations and mutable values."""

        digest = hashlib.sha256()
        digest.update(self.edge_indices.astype("<i8", copy=False).tobytes())
        digest.update(self.multipliers.astype("<f4", copy=False).tobytes())
        return digest.hexdigest()
