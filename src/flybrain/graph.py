"""Compact directed sparse representation of a canonical connectome."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse import csr_array


def validate_sparse_edge_overlay(
    edge_indices: NDArray[np.int64],
    edge_multipliers: NDArray[np.float32],
    edge_count: int,
) -> None:
    """Reject malformed, nonfinite, or sign-reversing sparse edge overlays."""

    if (
        not isinstance(edge_indices, np.ndarray)
        or not isinstance(edge_multipliers, np.ndarray)
        or edge_indices.ndim != 1
        or edge_multipliers.ndim != 1
        or edge_indices.shape != edge_multipliers.shape
        or not np.issubdtype(edge_indices.dtype, np.integer)
        or not np.issubdtype(edge_multipliers.dtype, np.floating)
        or not np.all(edge_indices[:-1] < edge_indices[1:])
        or (edge_indices.size and (
            int(edge_indices[0]) < 0 or int(edge_indices[-1]) >= edge_count
        ))
        or not np.all(np.isfinite(edge_multipliers))
        or np.any(edge_multipliers < 0.0)
    ):
        raise ValueError("sparse edge overlay indices or multipliers are malformed")


@dataclass(frozen=True)
class SparseConnectome:
    """Neuron metadata plus a post-by-pre weighted sparse adjacency."""

    neuron_ids: NDArray[np.uint64]
    cell_types: tuple[str, ...]
    roles: tuple[str, ...]
    adjacency: csr_array
    transmitters: tuple[str, ...] = ()
    superclasses: tuple[str, ...] = ()

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
            transmitters=tuple(neurons.column("transmitter").to_pylist()),
            superclasses=tuple(neurons.column("superclass").to_pylist()),
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

    @property
    def storage_bytes(self) -> int:
        """Bytes used by CSR values, indices, and row pointers."""

        return int(
            self.adjacency.data.nbytes
            + self.adjacency.indices.nbytes
            + self.adjacency.indptr.nbytes
        )

    def propagate(self, activity: NDArray[np.float32]) -> NDArray[np.float32]:
        """Propagate presynaptic activity to postsynaptic weighted input."""

        if activity.shape != (self.neuron_count,):
            expected = (self.neuron_count,)
            raise ValueError(f"expected activity shape {expected}, got {activity.shape}")
        return np.asarray(self.adjacency @ activity, dtype=np.float32)


@dataclass(frozen=True)
class EventConnectome:
    """Pre-by-post CSR optimized for propagating sparse spike events."""

    neuron_ids: NDArray[np.uint64]
    cell_types: tuple[str, ...]
    roles: tuple[str, ...]
    transmitters: tuple[str, ...]
    superclasses: tuple[str, ...]
    outgoing: csr_array

    @classmethod
    def from_sparse(cls, graph: SparseConnectome) -> EventConnectome:
        outgoing = graph.adjacency.transpose().tocsr()
        outgoing.sum_duplicates()
        outgoing.sort_indices()
        return cls(
            neuron_ids=graph.neuron_ids,
            cell_types=graph.cell_types,
            roles=graph.roles,
            transmitters=graph.transmitters,
            superclasses=graph.superclasses,
            outgoing=outgoing,
        )

    @property
    def neuron_count(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.outgoing.nnz)

    @property
    def storage_items(self) -> int:
        return int(
            self.outgoing.data.size
            + self.outgoing.indices.size
            + self.outgoing.indptr.size
        )

    @property
    def storage_bytes(self) -> int:
        return int(
            self.outgoing.data.nbytes
            + self.outgoing.indices.nbytes
            + self.outgoing.indptr.nbytes
        )

    def propagate_indices(
        self,
        fired_indices: NDArray[np.int64],
        *,
        scale: float | NDArray[np.float32],
        edge_indices: NDArray[np.int64] | None = None,
        edge_multipliers: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Accumulate outgoing signed weights for only the neurons that fired.

        ``edge_indices`` and ``edge_multipliers`` provide an optional sparse
        overlay over the immutable CSR data.  The indices are positions in
        ``outgoing.data``; applying them here avoids copying the entire graph
        for every neural integration window.
        """

        if fired_indices.ndim != 1:
            raise ValueError("fired_indices must be one-dimensional")
        if fired_indices.size and (
            int(fired_indices.min()) < 0 or int(fired_indices.max()) >= self.neuron_count
        ):
            raise ValueError("fired index outside graph")
        if isinstance(scale, np.ndarray) and scale.shape != fired_indices.shape:
            raise ValueError("per-source scale must match fired indices")
        if (edge_indices is None) != (edge_multipliers is None):
            raise ValueError("edge indices and multipliers must be provided together")
        if edge_indices is not None and edge_multipliers is not None:
            validate_sparse_edge_overlay(
                edge_indices, edge_multipliers, self.outgoing.data.size
            )

        def overlay_for_positions(positions: NDArray[np.int64]) -> NDArray[np.float32]:
            if edge_indices is None or edge_multipliers is None:
                return np.ones(positions.size, dtype=np.float32)
            locations = np.searchsorted(edge_indices, positions)
            matches = (locations < edge_indices.size) & (
                edge_indices[locations.clip(max=max(edge_indices.size - 1, 0))]
                == positions
            ) if edge_indices.size else np.zeros(positions.size, dtype=np.bool_)
            values = np.ones(positions.size, dtype=np.float32)
            values[matches] = edge_multipliers[locations[matches]]
            return values

        if fired_indices.size >= 64:
            selected = self.outgoing[fired_indices]
            if edge_indices is not None and edge_multipliers is not None:
                selected = selected.tocsr(copy=True)
                positions = np.concatenate(
                    tuple(
                        np.arange(
                            int(self.outgoing.indptr[index]),
                            int(self.outgoing.indptr[index + 1]),
                            dtype=np.int64,
                        )
                        for index in fired_indices
                    )
                )
                selected.data *= overlay_for_positions(positions)
            scales = (
                np.full(fired_indices.size, scale, dtype=np.float32)
                if np.isscalar(scale)
                else np.asarray(scale, dtype=np.float32)
            )
            return np.asarray(selected.T @ scales, dtype=np.float32).ravel()

        output = np.zeros(self.neuron_count, dtype=np.float32)
        for position, pre_index in enumerate(fired_indices):
            start = self.outgoing.indptr[pre_index]
            stop = self.outgoing.indptr[pre_index + 1]
            postsynaptic = self.outgoing.indices[start:stop]
            source_scale = scale[position] if isinstance(scale, np.ndarray) else scale
            values = self.outgoing.data[start:stop] * np.float32(source_scale)
            if edge_indices is not None and edge_multipliers is not None:
                values *= overlay_for_positions(np.arange(start, stop, dtype=np.int64))
            np.add.at(output, postsynaptic, values)
        return output
