"""Closed-loop execution joining sparse neural dynamics to a bounded world."""

from __future__ import annotations

import hashlib
import math
import resource
import sys
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.embodied_interfaces import (
    ExternalEvent,
    MotorDecoder,
    MotorMap,
    SensoryEncoder,
    SensoryMap,
)
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody, MotorCommand
from flybrain.graph import EventConnectome
from flybrain.plastic_graph import materialize_plastic_event_graph
from flybrain.plasticity import PlasticEdgeSet, PlasticityParameters
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu


class EmbodiedEpisodeConfig(BaseModel, frozen=True):
    """Bounded deterministic episode settings and declared neuron maps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_steps: int
    chunk_steps: int = 1
    seed: int = 7
    sensory_map: SensoryMap
    motor_map: MotorMap
    parameters: ShiuParameters = ShiuParameters()

    def validate_episode(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        self.parameters.validate()


class EmbodiedEpisodeResult(BaseModel, frozen=True):
    """Auditable traces and gates from one closed-loop episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark: str
    interface_evidence: Literal["arbitrary-smoke-only"]
    behavior_claim_allowed: Literal[False]
    snapshot: str
    dataset_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int
    graph_edges: int
    steps: int
    neural_steps: int
    sensory_map: SensoryMap
    motor_map: MotorMap
    sensory_events: tuple[ExternalEvent, ...]
    actions: tuple[MotorCommand, ...]
    rewards: tuple[float, ...]
    dopamine: tuple[float, ...]
    body_trace: tuple[FlyBody, ...]
    trace_digest: str
    replay_exact: bool
    graph_unchanged: bool
    executed_graph_unchanged: bool
    motor_silenced: bool
    learning_applied: bool
    final_plastic_multipliers: tuple[float, ...]
    passed: bool
    software_revision: str
    runtime_seconds: float
    peak_rss_bytes: int


@dataclass(frozen=True)
class _EpisodeTrace:
    sensory_events: tuple[ExternalEvent, ...]
    actions: tuple[MotorCommand, ...]
    rewards: tuple[float, ...]
    dopamine: tuple[float, ...]
    body_trace: tuple[FlyBody, ...]
    spikes: tuple[tuple[int, ...], ...]
    final_plastic_multipliers: tuple[float, ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


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


def _software_revision() -> str:
    from flybrain.mb_association import _software_revision as revision

    return revision()


def _clone_world(world: ArenaWorld, body: FlyBody) -> ArenaWorld:
    return ArenaWorld(
        world.config,
        body,
        food=world.food,
        threat=world.threat,
        wall_segments=world.wall_segments,
    )


def _clone_plastic_edges(edges: PlasticEdgeSet | None) -> PlasticEdgeSet | None:
    if edges is None:
        return None
    return PlasticEdgeSet(
        pre_ids=edges.pre_ids.copy(),
        post_ids=edges.post_ids.copy(),
        baseline_weights=edges.baseline_weights.copy(),
        multipliers=edges.multipliers.copy(),
        eligibility=edges.eligibility.copy(),
    )


def _validate_ids(graph: EventConnectome, config: EmbodiedEpisodeConfig) -> None:
    available = {int(value) for value in graph.neuron_ids}
    declared = (
        *config.sensory_map.visual_ids,
        *config.sensory_map.odor_ids,
        *config.sensory_map.touch_ids,
        *config.sensory_map.proprioception_ids,
        *config.motor_map.left_ids,
        *config.motor_map.right_ids,
        *config.motor_map.forward_ids,
    )
    missing = sorted(set(declared) - available)
    if missing:
        raise ValueError(f"declared embodied neuron IDs are absent from graph: {missing}")


def _execute(
    graph: EventConnectome,
    config: EmbodiedEpisodeConfig,
    world: ArenaWorld,
    *,
    silence_motor: bool,
    plastic_edges: PlasticEdgeSet | None,
    plasticity_params: PlasticityParameters,
) -> _EpisodeTrace:
    params = config.parameters
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    encoder = SensoryEncoder(
        config.sensory_map,
        arena_width=world.config.width,
        arena_height=world.config.height,
    )
    decoder = MotorDecoder(config.motor_map)
    state = ShiuState.initial(graph.neuron_count, params=params, seed=config.seed)
    sensory_indices = np.asarray(
        [index_by_id[int(value)] for value in (
            *config.sensory_map.visual_ids,
            *config.sensory_map.odor_ids,
            *config.sensory_map.touch_ids,
            *config.sensory_map.proprioception_ids,
        )],
        dtype=np.int64,
    )
    refractory_exempt = np.zeros(graph.neuron_count, dtype=np.bool_)
    refractory_exempt[sensory_indices] = True
    silence_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
    if silence_motor:
        for neuron_id in (
            *config.motor_map.left_ids,
            *config.motor_map.right_ids,
            *config.motor_map.forward_ids,
        ):
            silence_mask[index_by_id[neuron_id]] = True
    sensory_events: list[ExternalEvent] = []
    actions: list[MotorCommand] = []
    rewards: list[float] = []
    dopamine: list[float] = []
    body_trace: list[FlyBody] = []
    spikes_trace: list[tuple[int, ...]] = []
    active_edges = _clone_plastic_edges(plastic_edges)
    for step in range(config.max_steps):
        encoded = encoder.encode(world.observe(), step=step)
        sensory_events.extend(encoded)
        indices: list[int] = []
        voltages: list[float] = []
        for event in encoded:
            indices.extend(index_by_id[neuron_id] for neuron_id in event.neuron_ids)
            voltages.extend(event.voltages)
        external = (
            {
                state.step + offset: (
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(voltages, dtype=np.float32),
                )
                for offset in range(config.chunk_steps)
            }
            if indices
            else {}
        )
        active_graph = graph
        if active_edges is not None:
            active_graph, _ = materialize_plastic_event_graph(graph, active_edges)
        batches = simulate_shiu(
            active_graph,
            params,
            steps=config.chunk_steps,
            external_voltage_events=external,
            seed=config.seed,
            state=state,
            silenced=silence_mask,
            refractory_exempt=refractory_exempt,
        )
        fired_ids = tuple(int(value) for batch in batches for value in batch.neuron_ids)
        spikes_trace.append(fired_ids)
        command = decoder.decode(fired_ids)
        actions.append(command)
        result = world.step(command)
        body_trace.append(result.body)
        reward = float(result.food_contact) - float(result.threat_contact)
        rewards.append(reward)
        dopamine.append(reward)
        if active_edges is not None:
            fired = np.asarray(fired_ids, dtype=np.uint64)
            active_edges.update_eligibility(
                active_pre_ids=fired,
                gated_post_ids=fired,
                dt_ms=config.chunk_steps * params.dt_ms,
                params=plasticity_params,
            )
            if reward != 0.0:
                active_edges.apply_dopamine(
                    {int(post_id): reward for post_id in np.unique(active_edges.post_ids)},
                    plasticity_params,
                )
    trace = _EpisodeTrace(
        sensory_events=tuple(sensory_events),
        actions=tuple(actions),
        rewards=tuple(rewards),
        dopamine=tuple(dopamine),
        body_trace=tuple(body_trace),
        spikes=tuple(spikes_trace),
        final_plastic_multipliers=(
            tuple(float(value) for value in active_edges.multipliers)
            if active_edges is not None
            else ()
        ),
    )
    if not all(math.isfinite(value) for value in trace.rewards):
        raise ValueError("episode produced non-finite rewards")
    return trace


def run_embodied_episode(
    graph: EventConnectome,
    config: EmbodiedEpisodeConfig,
    *,
    plastic_graph: EventConnectome | None = None,
    plastic_edges: PlasticEdgeSet | None = None,
    plasticity_params: PlasticityParameters | None = None,
    world: ArenaWorld | None = None,
    silence_motor: bool = False,
    snapshot: str = "unbound",
    dataset_id: str = "unbound",
    source_manifest_sha256: str = "0" * 64,
    snapshot_content_sha256: str = "0" * 64,
) -> EmbodiedEpisodeResult:
    """Run a deterministic closed loop and an independent deterministic replay."""

    started = time.perf_counter()
    config.validate_episode()
    _validate_ids(graph, config)
    active_graph = plastic_graph or graph
    if plastic_graph is not None and plastic_edges is not None:
        raise ValueError("plastic_graph and plastic_edges are mutually exclusive")
    _validate_ids(active_graph, config)
    initial_world = world or ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(8.0, 5.0),
        threat=(1.0, 1.0),
    )
    initial_body = initial_world.body
    graph_digest = _graph_digest(graph)
    active_graph_digest = _graph_digest(active_graph)
    active_plasticity_params = plasticity_params or PlasticityParameters()
    active_plasticity_params.validate()
    first = _execute(
        graph=active_graph,
        config=config,
        world=initial_world,
        silence_motor=silence_motor,
        plastic_edges=plastic_edges,
        plasticity_params=active_plasticity_params,
    )
    replay_world = _clone_world(initial_world, initial_body)
    replay = _execute(
        graph=active_graph,
        config=config,
        world=replay_world,
        silence_motor=silence_motor,
        plastic_edges=plastic_edges,
        plasticity_params=active_plasticity_params,
    )
    replay_exact = first == replay
    graph_unchanged = _graph_digest(graph) == graph_digest
    executed_graph_unchanged = _graph_digest(active_graph) == active_graph_digest
    passed = bool(
        first.sensory_events
        and len(first.actions) == config.max_steps
        and replay_exact
        and graph_unchanged
        and executed_graph_unchanged
    )
    return EmbodiedEpisodeResult(
        benchmark="embodied-loop-v1",
        interface_evidence="arbitrary-smoke-only",
        behavior_claim_allowed=False,
        snapshot=snapshot,
        dataset_id=dataset_id,
        source_manifest_sha256=source_manifest_sha256,
        snapshot_content_sha256=snapshot_content_sha256,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        steps=config.max_steps,
        neural_steps=config.max_steps * config.chunk_steps,
        sensory_map=config.sensory_map,
        motor_map=config.motor_map,
        sensory_events=first.sensory_events,
        actions=first.actions,
        rewards=first.rewards,
        dopamine=first.dopamine,
        body_trace=first.body_trace,
        trace_digest=first.digest,
        replay_exact=replay_exact,
        graph_unchanged=graph_unchanged,
        executed_graph_unchanged=executed_graph_unchanged,
        motor_silenced=silence_motor,
        learning_applied=(
            plastic_edges is not None
            and first.final_plastic_multipliers
            != tuple(float(value) for value in plastic_edges.multipliers)
        ),
        final_plastic_multipliers=first.final_plastic_multipliers,
        passed=passed,
        software_revision=_software_revision(),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
