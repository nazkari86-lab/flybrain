"""Causal associative-plasticity calibration with explicit nonbehavioral claims."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.conditioning_world import ConditioningSchedule
from flybrain.graph import EventConnectome
from flybrain.mushroom_body_learning import (
    MushroomBodyLearning,
    MushroomBodyLearningParameters,
)
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_edge_registry import ResolvedPlasticEdgeManifest
from flybrain.reinforcement_interface import AnonymousContact, ReinforcementInterface
from flybrain.shiu import ShiuParameters, ShiuState, simulate_shiu


class AssociativeCalibrationConfig(BaseModel, frozen=True):
    """Predeclared calibration-only interfaces; no motor or world-command input exists."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cue_ids: tuple[int, ...]
    mbon_ids: tuple[int, ...]
    dan_to_mbon_pairs: tuple[tuple[int, int], ...]
    input_mode: Literal["direct_kc", "sensory_path"] = "direct_kc"
    sensory_input_ids: tuple[int, ...] = ()
    neural_chunk_steps: int = Field(default=20, gt=0)
    cue_voltage_mv: float = Field(default=100.0, gt=0.0)
    shiu_parameters: ShiuParameters = ShiuParameters()
    presynaptic_transmitter_multipliers: dict[str, float] = Field(default_factory=dict)
    learning_parameters: MushroomBodyLearningParameters = (
        MushroomBodyLearningParameters()
    )

    @classmethod
    def from_manifest(
        cls,
        binding: PlasticEdgeBinding,
        dan_to_mbon: ResolvedPlasticEdgeManifest,
        reinforcement: ReinforcementInterface,
        *,
        valence: Literal["appetitive", "aversive"],
        sensory_input_ids: tuple[int, ...] = (),
        neural_chunk_steps: int = 20,
        cue_voltage_mv: float = 100.0,
        shiu_parameters: ShiuParameters | None = None,
        presynaptic_transmitter_multipliers: dict[str, float] | None = None,
        learning_parameters: MushroomBodyLearningParameters | None = None,
    ) -> AssociativeCalibrationConfig:
        """Build calibration routes only from measured sign-zero DAN adjacency."""

        if dan_to_mbon.name != "dan_to_mbon" or dan_to_mbon.sign != 0:
            raise ValueError("calibration requires the declared sign-zero DAN-to-MBON manifest")
        permitted_dan_ids = (
            reinforcement.appetitive_dan_ids
            if valence == "appetitive"
            else reinforcement.aversive_dan_ids
        )
        mbon_ids = tuple(sorted(set(int(value) for value in binding.post_ids)))
        routes = tuple(
            (pre_id, post_id)
            for pre_id, post_id in dan_to_mbon.edge_pairs
            if pre_id in permitted_dan_ids and post_id in mbon_ids
        )
        if not routes:
            raise ValueError("no declared DAN-to-MBON routes for requested valence")
        return cls(
            cue_ids=tuple(sorted(set(int(value) for value in binding.pre_ids))),
            mbon_ids=mbon_ids,
            dan_to_mbon_pairs=routes,
            input_mode="sensory_path" if sensory_input_ids else "direct_kc",
            sensory_input_ids=sensory_input_ids,
            neural_chunk_steps=neural_chunk_steps,
            cue_voltage_mv=cue_voltage_mv,
            shiu_parameters=shiu_parameters or ShiuParameters(),
            presynaptic_transmitter_multipliers=(
                presynaptic_transmitter_multipliers or {}
            ),
            learning_parameters=learning_parameters or MushroomBodyLearningParameters(),
        )

    @model_validator(mode="after")
    def validate_interfaces(self) -> AssociativeCalibrationConfig:
        for name, ids in (("cue", self.cue_ids), ("MBON", self.mbon_ids)):
            if not ids or any(type(neuron_id) is not int or neuron_id <= 0 for neuron_id in ids):
                raise ValueError(f"{name} IDs must be positive and nonempty")
            if tuple(sorted(set(ids))) != ids:
                raise ValueError(f"{name} IDs must be unique and sorted")
        if not self.dan_to_mbon_pairs:
            raise ValueError("DAN-to-MBON routes must be nonempty")
        if self.input_mode == "direct_kc" and self.sensory_input_ids:
            raise ValueError("direct KC calibration cannot declare sensory input IDs")
        if self.input_mode == "sensory_path":
            if not self.sensory_input_ids or any(
                type(neuron_id) is not int or neuron_id <= 0
                for neuron_id in self.sensory_input_ids
            ):
                raise ValueError("sensory-path calibration requires positive input IDs")
            if tuple(sorted(set(self.sensory_input_ids))) != self.sensory_input_ids:
                raise ValueError("sensory input IDs must be unique and sorted")
            if set(self.sensory_input_ids) & set(self.cue_ids):
                raise ValueError("sensory input IDs must not bypass into declared KC IDs")
        if any(
            type(dan_id) is not int
            or type(mbon_id) is not int
            or dan_id <= 0
            or mbon_id not in self.mbon_ids
            for dan_id, mbon_id in self.dan_to_mbon_pairs
        ):
            raise ValueError("DAN-to-MBON routes must target declared MBON IDs")
        self.shiu_parameters.validate()
        if any(
            not isinstance(name, str) or not np.isfinite(value) or value < 0.0
            for name, value in self.presynaptic_transmitter_multipliers.items()
        ):
            raise ValueError(
                "transmitter multipliers must have string names and finite values >= 0"
            )
        self.learning_parameters.validate()
        return self


class AssociativeCalibrationResult(BaseModel, frozen=True):
    """A local-plasticity result explicitly separated from behavioral learning claims."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal["plasticity_calibration", "null"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    input_mode: Literal["direct_kc", "sensory_path"]
    schedule_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_exact: bool
    graph_unchanged: bool
    cue_spikes: int = Field(ge=0)
    mbon_spikes: int = Field(ge=0)
    contact_dan_events: int = Field(ge=0)
    final_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class _CalibrationTrace:
    cue_spikes: int
    mbon_spikes: int
    contact_dan_events: int
    final_multipliers: tuple[float, ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


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


def _clone_binding(binding: PlasticEdgeBinding) -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=binding.overlay.copy(),
        pre_ids=binding.pre_ids.copy(),
        post_ids=binding.post_ids.copy(),
    )


def _effective_graph(graph: EventConnectome, binding: PlasticEdgeBinding) -> EventConnectome:
    if binding.overlay.canonical_edge_count != graph.edge_count:
        raise ValueError("plastic binding does not match canonical graph edge count")
    outgoing = graph.outgoing.copy()
    outgoing.data = binding.overlay.apply_to(graph.outgoing.data)
    return EventConnectome(
        neuron_ids=graph.neuron_ids,
        cell_types=graph.cell_types,
        roles=graph.roles,
        transmitters=graph.transmitters,
        superclasses=graph.superclasses,
        outgoing=outgoing,
    )


def _execute(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    schedule: ConditioningSchedule,
    reinforcement: ReinforcementInterface,
    config: AssociativeCalibrationConfig,
) -> _CalibrationTrace:
    available = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    required = {
        *(config.cue_ids if config.input_mode == "direct_kc" else config.sensory_input_ids),
        *config.mbon_ids,
        *(dan_id for dan_id, _ in config.dan_to_mbon_pairs),
        *binding.pre_ids.tolist(),
        *binding.post_ids.tolist(),
    }
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"calibration IDs absent from graph: {missing}")
    if set(binding.pre_ids.tolist()) - set(config.cue_ids):
        raise ValueError("plastic KC endpoints must be declared cue IDs for calibration")
    if set(binding.post_ids.tolist()) - set(config.mbon_ids):
        raise ValueError("plastic MBON endpoints must be declared calibration MBON IDs")

    state = ShiuState.initial(graph.neuron_count, params=config.shiu_parameters, seed=schedule.seed)
    learning = MushroomBodyLearning(
        overlay=binding.overlay,
        edge_pre_ids=binding.pre_ids,
        edge_post_ids=binding.post_ids,
        dan_ids=np.asarray([item[0] for item in config.dan_to_mbon_pairs], dtype=np.uint64),
        dan_post_ids=np.asarray([item[1] for item in config.dan_to_mbon_pairs], dtype=np.uint64),
        parameters=config.learning_parameters,
    )
    cue_spikes = 0
    mbon_spikes = 0
    contact_dan_events = 0
    source_ids = (
        config.cue_ids if config.input_mode == "direct_kc" else config.sensory_input_ids
    )
    source_indices = np.asarray([available[item] for item in source_ids], dtype=np.int64)
    for event in schedule.events:
        external = {}
        if event.odor_intensity:
            external[state.step] = (
                source_indices,
                np.full(
                    source_indices.size,
                    config.cue_voltage_mv * event.odor_intensity,
                    dtype=np.float32,
                ),
            )
        batches = tuple(
            simulate_shiu(
                _effective_graph(graph, binding),
                config.shiu_parameters,
                steps=config.neural_chunk_steps,
                external_voltage_events=external,
                seed=schedule.seed,
                state=state,
                presynaptic_transmitter_multipliers=(
                    config.presynaptic_transmitter_multipliers
                ),
            )
        )
        fired_ids = np.asarray(
            [neuron_id for batch in batches for neuron_id in batch.neuron_ids], dtype=np.uint64
        )
        cue_spikes += int(np.isin(fired_ids, config.cue_ids).sum())
        mbon_spikes += int(np.isin(fired_ids, config.mbon_ids).sum())
        recruitment = reinforcement.recruit(
            AnonymousContact(
                appetitive_intensity=event.appetitive_contact_intensity,
                aversive_intensity=event.aversive_contact_intensity,
            )
        )
        if recruitment.dan_ids:
            contact_dan_events += 1
        learning.step(
            active_kc_ids=fired_ids[np.isin(fired_ids, config.cue_ids)],
            active_mbon_ids=fired_ids[np.isin(fired_ids, config.mbon_ids)],
            routed_dan_ids=np.asarray(recruitment.dan_ids, dtype=np.uint64),
            dt_ms=config.neural_chunk_steps * config.shiu_parameters.dt_ms,
        )
    return _CalibrationTrace(
        cue_spikes=cue_spikes,
        mbon_spikes=mbon_spikes,
        contact_dan_events=contact_dan_events,
        final_multipliers=tuple(float(value) for value in binding.overlay.multipliers),
    )


def run_associative_calibration(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    *,
    schedule: ConditioningSchedule,
    reinforcement: ReinforcementInterface,
    config: AssociativeCalibrationConfig,
) -> AssociativeCalibrationResult:
    """Run a replayable calibration; it makes no body or autonomous-behavior claim."""

    graph_digest = _graph_digest(graph)
    first = _execute(
        graph,
        _clone_binding(binding),
        schedule=schedule,
        reinforcement=reinforcement,
        config=config,
    )
    replay = _execute(
        graph,
        _clone_binding(binding),
        schedule=schedule,
        reinforcement=reinforcement,
        config=config,
    )
    changed = any(value != 1.0 for value in first.final_multipliers)
    return AssociativeCalibrationResult(
        classification="plasticity_calibration" if changed else "null",
        schedule_digest=schedule.digest,
        trace_digest=first.digest,
        replay_exact=first == replay,
        graph_unchanged=_graph_digest(graph) == graph_digest,
        input_mode=config.input_mode,
        cue_spikes=first.cue_spikes,
        mbon_spikes=first.mbon_spikes,
        contact_dan_events=first.contact_dan_events,
        final_multipliers=first.final_multipliers,
    )
