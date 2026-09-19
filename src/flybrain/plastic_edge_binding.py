"""Bind an immutable edge manifest to exact sparse CSR locations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flybrain.graph import EventConnectome
from flybrain.plastic_edge_registry import ResolvedPlasticEdgeManifest
from flybrain.plastic_overlay import PlasticWeightOverlay


@dataclass(frozen=True)
class PlasticEdgeBinding:
    """Canonical CSR locations and anatomical endpoints for one plastic edge manifest."""

    overlay: PlasticWeightOverlay
    pre_ids: NDArray[np.uint64]
    post_ids: NDArray[np.uint64]


def bind_manifest_to_graph(
    graph: EventConnectome,
    manifest: ResolvedPlasticEdgeManifest,
) -> PlasticEdgeBinding:
    """Fail closed unless each declared pair maps once to a canonical CSR edge."""

    if manifest.sign != 1:
        raise ValueError("only positive KC-to-MBON edges may bind a plastic overlay")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    locations: list[int] = []
    for pre_id, post_id in manifest.edge_pairs:
        try:
            pre_index = index_by_id[pre_id]
            post_index = index_by_id[post_id]
        except KeyError as error:
            raise ValueError("plastic manifest neuron absent from graph") from error
        start = int(graph.outgoing.indptr[pre_index])
        stop = int(graph.outgoing.indptr[pre_index + 1])
        matches = np.flatnonzero(graph.outgoing.indices[start:stop] == post_index)
        if matches.size != 1:
            raise ValueError("plastic manifest pair does not map once to graph")
        locations.append(start + int(matches[0]))
    indices = np.asarray(locations, dtype=np.int64)
    if not np.all(indices[:-1] < indices[1:]):
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        pairs = tuple(manifest.edge_pairs[index] for index in order)
    else:
        pairs = manifest.edge_pairs
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=indices,
            canonical_edge_count=graph.edge_count,
        ),
        pre_ids=np.asarray([pair[0] for pair in pairs], dtype=np.uint64),
        post_ids=np.asarray([pair[1] for pair in pairs], dtype=np.uint64),
    )
