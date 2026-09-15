"""Explicit causal interventions on canonical neuron populations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flybrain.graph import SparseConnectome


@dataclass(frozen=True)
class NeuronRecord:
    """Stable metadata exposed to an intervention predicate."""

    neuron_id: int
    cell_type: str
    role: str


def silence_mask(
    graph: SparseConnectome,
    predicate: Callable[[NeuronRecord], bool],
) -> NDArray[np.bool_]:
    """Build an index-aligned mask for neurons selected by metadata."""

    return np.fromiter(
        (
            predicate(NeuronRecord(int(neuron_id), cell_type, role))
            for neuron_id, cell_type, role in zip(
                graph.neuron_ids, graph.cell_types, graph.roles, strict=True
            )
        ),
        dtype=np.bool_,
        count=graph.neuron_count,
    )
