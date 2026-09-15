"""Compact directed sparse representation of a canonical connectome."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse import csr_array


@dataclass(frozen=True)
class SparseConnectome:
    """Neuron metadata plus a post-by-pre weighted sparse adjacency."""

    neuron_ids: NDArray[np.uint64]
    cell_types: tuple[str, ...]
    roles: tuple[str, ...]
    adjacency: csr_array

    @classmethod
    def from_snapshot(cls, path: Path) -> SparseConnectome:
        neurons = pq.read_table(path / "neurons.parquet").sort_by("neuron_id")
        edges = pq.read_table(path / "edges.parquet")

        neuron_ids = neurons.column("neuron_id").to_numpy().astype(np.uint64, copy=False)
        index = {int(neuron_id): position for position, neuron_id in enumerate(neuron_ids)}
        pre = np.fromiter(
            (index[int(value)] for value in edges.column("pre_id").to_pylist()), dtype=np.int64
        )
        post = np.fromiter(
            (index[int(value)] for value in edges.column("post_id").to_pylist()), dtype=np.int64
        )
        counts = edges.column("synapse_count").to_numpy().astype(np.float32, copy=False)
        signs = edges.column("sign").to_numpy().astype(np.float32, copy=False)
        weights = counts * signs
        adjacency = csr_array((weights, (post, pre)), shape=(len(neuron_ids), len(neuron_ids)))
        adjacency.sum_duplicates()
        adjacency.sort_indices()

        return cls(
            neuron_ids=neuron_ids,
            cell_types=tuple(neurons.column("cell_type").to_pylist()),
            roles=tuple(neurons.column("role").to_pylist()),
            adjacency=adjacency,
        )

    @property
    def neuron_count(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.adjacency.nnz)

    @property
    def storage_items(self) -> int:
        return int(
            self.adjacency.data.size
            + self.adjacency.indices.size
            + self.adjacency.indptr.size
        )

    def propagate(self, activity: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate presynaptic activity to postsynaptic weighted input."""

        if activity.shape != (self.neuron_count,):
            expected = (self.neuron_count,)
            raise ValueError(f"expected activity shape {expected}, got {activity.shape}")
        return np.asarray(self.adjacency @ activity, dtype=np.float32)
