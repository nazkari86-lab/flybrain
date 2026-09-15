"""Sparse materialization of persistent plastic weights into an event graph."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from flybrain.graph import EventConnectome
from flybrain.plasticity import PlasticEdgeSet


class PlasticGraphMetrics(BaseModel):
    """Accounting for a materialized plastic edge overlay."""

    model_config = ConfigDict(frozen=True)

    matched_edges: int
    modified_edges: int
    baseline_absolute_weight: float
    effective_absolute_weight: float
    min_multiplier: float
    max_multiplier: float


def materialize_plastic_event_graph(
    graph: EventConnectome,
    edges: PlasticEdgeSet,
) -> tuple[EventConnectome, PlasticGraphMetrics]:
    """Copy an event graph and apply plastic multipliers to selected CSR entries."""

    pairs = [
        (int(pre_id), int(post_id))
        for pre_id, post_id in zip(edges.pre_ids, edges.post_ids, strict=True)
    ]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate plastic edge pair")
    if not pairs:
        raise ValueError("at least one plastic edge is required")
    if not np.all(np.isfinite(edges.multipliers)):
        raise ValueError("plastic edge multipliers must be finite")

    neuron_index = {
        int(neuron_id): position for position, neuron_id in enumerate(graph.neuron_ids)
    }
    copied = graph.outgoing.copy()
    baseline_absolute_weight = 0.0
    effective_absolute_weight = 0.0

    for edge_index, (pre_id, post_id) in enumerate(pairs):
        if pre_id not in neuron_index:
            raise ValueError(f"unknown neuron ID {pre_id}")
        if post_id not in neuron_index:
            raise ValueError(f"unknown neuron ID {post_id}")

        pre_index = neuron_index[pre_id]
        post_index = neuron_index[post_id]
        start = copied.indptr[pre_index]
        stop = copied.indptr[pre_index + 1]
        row_indices = copied.indices[start:stop]
        row_position = int(np.searchsorted(row_indices, post_index))
        if row_position == row_indices.size or int(row_indices[row_position]) != post_index:
            raise ValueError(f"missing graph edge {pre_id}->{post_id}")
        if row_position + 1 < row_indices.size and int(row_indices[row_position + 1]) == post_index:
            raise ValueError(f"missing graph edge: {pre_id}->{post_id} is not unique")

        position = int(start) + row_position
        value = float(copied.data[position])
        if value == 0.0:
            raise ValueError(f"zero-sign graph edge {pre_id}->{post_id}")
        baseline = float(edges.baseline_weights[edge_index])
        if not np.isclose(abs(value), baseline, rtol=1e-6, atol=1e-6):
            raise ValueError(f"baseline weight mismatch for graph edge {pre_id}->{post_id}")

        multiplier = float(edges.multipliers[edge_index])
        effective = value * multiplier
        copied.data[position] = effective
        baseline_absolute_weight += abs(value)
        effective_absolute_weight += abs(effective)

    learned = EventConnectome(
        neuron_ids=graph.neuron_ids,
        cell_types=graph.cell_types,
        roles=graph.roles,
        transmitters=graph.transmitters,
        superclasses=graph.superclasses,
        outgoing=copied,
    )
    metrics = PlasticGraphMetrics(
        matched_edges=len(pairs),
        modified_edges=int(np.count_nonzero(edges.multipliers != 1.0)),
        baseline_absolute_weight=baseline_absolute_weight,
        effective_absolute_weight=effective_absolute_weight,
        min_multiplier=float(np.min(edges.multipliers)),
        max_multiplier=float(np.max(edges.multipliers)),
    )
    return learned, metrics
