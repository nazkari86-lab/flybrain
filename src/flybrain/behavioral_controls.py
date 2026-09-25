"""Independent biological control conditions for autonomous behavior assays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.graph import EventConnectome
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay

ControlCondition = Literal[
    "normal",
    "no_plasticity",
    "dan_lesion",
    "kc_mbon_lesion",
    "rewired_control",
]
CONDITIONS: tuple[ControlCondition, ...] = (
    "normal",
    "no_plasticity",
    "dan_lesion",
    "kc_mbon_lesion",
    "rewired_control",
)


@dataclass(frozen=True)
class ConditionBinding:
    """One condition's fully independent mutable state and routing flags."""

    condition: ControlCondition
    overlay: PlasticWeightOverlay
    learning: AssociativeCalibrationConfig
    dan_enabled: bool
    plasticity_enabled: bool
    rewired: bool
    pre_ids: tuple[int, ...]
    post_ids: tuple[int, ...]


@dataclass(frozen=True)
class RewiredPlasticGraph:
    """A degree-preserving structural control, isolated from canonical anatomy."""

    graph: EventConnectome
    binding: PlasticEdgeBinding
    changed_edges: int


def rewire_plastic_edges(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    seed: int,
) -> RewiredPlasticGraph:
    """Swap KC→MBON targets while preserving source and target edge degrees.

    Swaps are rejected if they would duplicate any canonical nonplastic edge or
    any current plastic edge.  An unsuccessful control is returned unchanged and
    must not count as evidence for a behavioral claim.
    """

    locations = binding.overlay.edge_indices
    if binding.overlay.canonical_edge_count != graph.edge_count:
        raise ValueError("rewired control binding does not match graph edge count")
    if binding.pre_ids.shape != locations.shape or binding.post_ids.shape != locations.shape:
        raise ValueError("rewired control edge endpoints do not match overlay")
    if locations.size == 0 or not graph.outgoing.has_sorted_indices:
        raise ValueError("rewired control requires nonempty sorted canonical CSR edges")
    if (
        not np.all(locations[:-1] < locations[1:])
        or int(locations[0]) < 0
        or int(locations[-1]) >= graph.edge_count
    ):
        raise ValueError("rewired control edge location outside graph")
    rows = np.searchsorted(graph.outgoing.indptr, locations, side="right") - 1
    original_posts = graph.outgoing.indices[locations]
    if not np.array_equal(graph.neuron_ids[rows], binding.pre_ids) or not np.array_equal(
        graph.neuron_ids[original_posts], binding.post_ids
    ):
        raise ValueError("rewired control binding endpoints disagree with canonical graph")
    original_pairs = set(zip(rows.tolist(), original_posts.tolist(), strict=True))
    if len(original_pairs) != locations.size:
        raise ValueError("rewired control plastic edges are not unique")
    current_pairs = original_pairs.copy()
    posts = original_posts.copy()
    generator = np.random.default_rng(seed)

    def occupied_nonplastic(row: int, post: int) -> bool:
        if (row, post) in original_pairs:
            return False
        start = int(graph.outgoing.indptr[row])
        stop = int(graph.outgoing.indptr[row + 1])
        neighbors = graph.outgoing.indices[start:stop]
        position = int(np.searchsorted(neighbors, post))
        return position < neighbors.size and int(neighbors[position]) == post

    for _ in range(max(16, 2 * locations.size)):
        first, second = generator.integers(0, locations.size, size=2)
        if first == second:
            continue
        first_row, second_row = int(rows[first]), int(rows[second])
        first_post, second_post = int(posts[first]), int(posts[second])
        if first_row == second_row or first_post == second_post:
            continue
        if second_post == int(original_posts[first]) or first_post == int(original_posts[second]):
            continue
        proposed = ((first_row, second_post), (second_row, first_post))
        if any(pair in current_pairs for pair in proposed):
            continue
        if any(occupied_nonplastic(*pair) for pair in proposed):
            continue
        current_pairs.remove((first_row, first_post))
        current_pairs.remove((second_row, second_post))
        current_pairs.update(proposed)
        posts[first], posts[second] = second_post, first_post

    changed = int(np.count_nonzero(posts != original_posts))
    if not np.array_equal(
        np.bincount(posts, minlength=graph.neuron_count),
        np.bincount(original_posts, minlength=graph.neuron_count),
    ):
        raise ValueError("rewired control failed to preserve target degrees")
    if changed == 0:
        return RewiredPlasticGraph(graph=graph, binding=binding, changed_edges=0)
    outgoing = graph.outgoing.copy()
    outgoing.indices[locations] = posts
    outgoing.has_sorted_indices = False
    rewired_graph = EventConnectome(
        neuron_ids=graph.neuron_ids,
        cell_types=graph.cell_types,
        roles=graph.roles,
        transmitters=graph.transmitters,
        superclasses=graph.superclasses,
        outgoing=outgoing,
    )
    return RewiredPlasticGraph(
        graph=rewired_graph,
        binding=PlasticEdgeBinding(
            overlay=binding.overlay,
            pre_ids=binding.pre_ids.copy(),
            post_ids=graph.neuron_ids[posts].astype(np.uint64, copy=True),
        ),
        changed_edges=changed,
    )


def build_condition(
    condition: ControlCondition,
    binding: PlasticEdgeBinding,
    learning: AssociativeCalibrationConfig,
    *,
    seed: int,
) -> ConditionBinding:
    """Construct an isolated control without editing canonical graph anatomy."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown control condition: {condition}")
    overlay = binding.overlay.copy()
    if condition == "no_plasticity":
        overlay.reset()
    elif condition == "kc_mbon_lesion":
        overlay.set_multipliers(
            np.zeros_like(overlay.multipliers), minimum=0.0, maximum=2.0
        )
    pre_ids = tuple(int(value) for value in binding.pre_ids)
    post_ids = tuple(int(value) for value in binding.post_ids)
    return ConditionBinding(
        condition=condition,
        overlay=overlay,
        learning=learning,
        dan_enabled=condition != "dan_lesion",
        plasticity_enabled=condition not in {"no_plasticity", "kc_mbon_lesion"},
        rewired=condition == "rewired_control",
        pre_ids=pre_ids,
        post_ids=post_ids,
    )
