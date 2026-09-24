"""Causal calibration of retained looming projection paths."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.graph import EventConnectome
from flybrain.retinal_interface import VisualLoomingMap
from flybrain.shiu import ShiuParameters, simulate_shiu


class LoomingConditionResult(BaseModel, frozen=True):
    """One fixed projection-stimulus condition and its endogenous DN response."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_events: int = Field(ge=0)
    source_neuron_events: int = Field(ge=0)
    dnp01_left_spikes: int = Field(ge=0)
    dnp01_right_spikes: int = Field(ge=0)
    dnp02_left_spikes: int = Field(ge=0)
    dnp02_right_spikes: int = Field(ge=0)
    dnp01_spikes: int = Field(ge=0)
    dnp02_spikes: int = Field(ge=0)
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    silenced_populations: tuple[str, ...]
    upstream_visual_processing_bypassed: Literal[True] = True


class VisualLoomingAssayResult(BaseModel, frozen=True):
    """Replayable projection-to-DN evidence, never an animal-behavior claim."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["visual-looming-projection-assay-v1"] = (
        "visual-looming-projection-assay-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(ge=0)
    snapshot: str = ""
    registry: dict[str, object] = Field(default_factory=dict)
    conditions: dict[str, LoomingConditionResult]
    causal_pathway_claim_allowed: bool
    behavioral_claim_allowed: Literal[False] = False
    replay_exact: bool
    graph_unchanged: bool


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for array in (
        graph.neuron_ids,
        graph.outgoing.data,
        graph.outgoing.indices,
        graph.outgoing.indptr,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _run_condition(
    graph: EventConnectome,
    mapping: VisualLoomingMap,
    *,
    name: str,
    steps: int,
    seed: int,
    parameters: ShiuParameters,
    silenced_populations: frozenset[str],
    stimulated: bool,
) -> LoomingConditionResult:
    populations = mapping.populations()
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    required = {neuron_id for ids in populations.values() for neuron_id in ids}
    missing = sorted(required - index_by_id.keys())
    if missing:
        raise ValueError(f"looming interface IDs absent from graph: {missing}")
    source_ids = (
        populations["lc4_left"]
        + populations["lc4_right"]
        + populations["lplc2_left"]
        + populations["lplc2_right"]
    )
    events = {
        step: (
            np.asarray([index_by_id[neuron_id] for neuron_id in source_ids], dtype=np.int64),
            np.full(len(source_ids), 68.75, dtype=np.float32),
        )
        for step in range(steps)
    } if stimulated else {}
    silence_ids = {
        neuron_id
        for population in silenced_populations
        for neuron_id in populations[population]
    }
    silence_mask = np.asarray(
        [int(neuron_id) in silence_ids for neuron_id in graph.neuron_ids], dtype=np.bool_
    )
    direct_mask = np.asarray(
        [int(neuron_id) in set(source_ids) for neuron_id in graph.neuron_ids], dtype=np.bool_
    )
    spike_trace: list[tuple[int, tuple[int, ...]]] = []
    counts = {name: 0 for name in populations if name.startswith("dnp")}
    for batch in simulate_shiu(
        graph,
        parameters,
        steps=steps,
        external_voltage_events=events,
        seed=seed,
        silenced=silence_mask,
        refractory_exempt=direct_mask,
    ):
        fired = tuple(int(neuron_id) for neuron_id in batch.neuron_ids)
        spike_trace.append((batch.step, fired))
        for population, ids in populations.items():
            if population.startswith("dnp"):
                counts[population] += sum(neuron_id in ids for neuron_id in fired)
    trace_digest = hashlib.sha256(
        json.dumps(spike_trace, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return LoomingConditionResult(
        name=name,
        source_events=len(events),
        source_neuron_events=sum(len(values[0]) for values in events.values()),
        dnp01_left_spikes=counts["dnp01_left"],
        dnp01_right_spikes=counts["dnp01_right"],
        dnp02_left_spikes=counts["dnp02_left"],
        dnp02_right_spikes=counts["dnp02_right"],
        dnp01_spikes=counts["dnp01_left"] + counts["dnp01_right"],
        dnp02_spikes=counts["dnp02_left"] + counts["dnp02_right"],
        trace_digest=trace_digest,
        silenced_populations=tuple(sorted(silenced_populations)),
    )


def run_visual_looming_assay(
    graph: EventConnectome,
    mapping: VisualLoomingMap,
    *,
    steps: int,
    seed: int,
    parameters: ShiuParameters | None = None,
) -> VisualLoomingAssayResult:
    """Measure retained LC4/LPLC2 to DNp01/DNp02 responses under lesions."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("looming assay steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("looming assay seed must be a non-negative integer")
    mapping.validate_disjoint_nonempty()
    before = _graph_digest(graph)
    active_parameters = parameters or ShiuParameters()
    conditions = {
        "static": _run_condition(
            graph,
            mapping,
            name="static",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset(),
            stimulated=False,
        ),
        "looming": _run_condition(
            graph,
            mapping,
            name="looming",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset(),
            stimulated=True,
        ),
        "lc4_lesion": _run_condition(
            graph,
            mapping,
            name="lc4_lesion",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset({"lc4_left", "lc4_right"}),
            stimulated=True,
        ),
        "lplc2_lesion": _run_condition(
            graph,
            mapping,
            name="lplc2_lesion",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset({"lplc2_left", "lplc2_right"}),
            stimulated=True,
        ),
        "lc4_lplc2_lesion": _run_condition(
            graph,
            mapping,
            name="lc4_lplc2_lesion",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset(
                {"lc4_left", "lc4_right", "lplc2_left", "lplc2_right"}
            ),
            stimulated=True,
        ),
        "dnp01_lesion": _run_condition(
            graph,
            mapping,
            name="dnp01_lesion",
            steps=steps,
            seed=seed,
            parameters=active_parameters,
            silenced_populations=frozenset({"dnp01_left", "dnp01_right"}),
            stimulated=True,
        ),
    }
    replay = _run_condition(
        graph,
        mapping,
        name="looming_replay",
        steps=steps,
        seed=seed,
        parameters=active_parameters,
        silenced_populations=frozenset(),
        stimulated=True,
    )
    normal = conditions["looming"]
    causal = (
        normal.dnp01_spikes > conditions["static"].dnp01_spikes
        and conditions["lc4_lplc2_lesion"].dnp01_spikes < normal.dnp01_spikes
        and replay.trace_digest == normal.trace_digest
    )
    return VisualLoomingAssayResult(
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        conditions={**conditions, "looming_replay": replay},
        causal_pathway_claim_allowed=causal,
        replay_exact=replay.trace_digest == normal.trace_digest,
        graph_unchanged=_graph_digest(graph) == before,
    )
