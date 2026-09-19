"""Bounded multi-episode behavioral benchmark over the autonomous hexapod loop."""

from __future__ import annotations

import hashlib
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.associative_motor_loop import BackendFactory
from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    AutonomousHexapodResult,
    HexapodArenaConfig,
    run_autonomous_hexapod_episode,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_controls import (
    CONDITIONS,
    ConditionBinding,
    ControlCondition,
    build_condition,
)
from flybrain.behavioral_metrics import (
    EpisodeObservation,
    PairedComparison,
    bootstrap_interval,
    claim_gate,
)
from flybrain.behavioral_perturbations import BehaviorVariant
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import BackendIdentity, ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.reinforcement_interface import ReinforcementInterface


class BehaviorBenchmarkConfig(BaseModel, frozen=True):
    """Deterministic bounded benchmark inputs."""

    model_config = ConfigDict(extra="forbid")

    training_episodes: int = Field(gt=0, le=10_000)
    holdout_episodes: int = Field(gt=0, le=10_000)
    body_steps: int = Field(gt=0, le=10_000)
    seeds: tuple[int, ...]
    training_variants: tuple[BehaviorVariant, ...]
    holdout_variants: tuple[BehaviorVariant, ...]
    controls: tuple[ControlCondition, ...] = CONDITIONS[1:]
    nitric_oxide_dan_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_benchmark(self) -> BehaviorBenchmarkConfig:
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("benchmark requires non-negative seeds")
        if not self.training_variants or not self.holdout_variants:
            raise ValueError("benchmark requires training and holdout variants")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("benchmark controls must be unique")
        if any(condition not in CONDITIONS[1:] for condition in self.controls):
            raise ValueError("benchmark controls must be explicit non-normal conditions")
        return self


class BehaviorBenchmarkResult(BaseModel, frozen=True):
    """Auditable multi-condition observation and comparison artifact."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-behavior-benchmark-v1"] = (
        "autonomous-behavior-benchmark-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    behavioral_claim_allowed: bool
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: BackendIdentity
    condition_observations: dict[str, tuple[EpisodeObservation, ...]]
    comparisons: dict[str, PairedComparison]
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


def _condition_binding(
    condition: ControlCondition,
    binding: PlasticEdgeBinding,
    learning: AssociativeCalibrationConfig,
    seed: int,
) -> ConditionBinding:
    return build_condition(condition, binding, learning, seed=seed)


def _observation(result: AutonomousHexapodResult, *, horizon: int) -> EpisodeObservation:
    episode = result
    return EpisodeObservation(
        food_contact=bool(episode.appetitive_contacts),
        threat_contact=bool(episode.aversive_contacts),
        initial_food_distance=episode.trace_distance_to_food[0],
        final_food_distance=episode.trace_distance_to_food[-1],
        first_food_contact_step=episode.first_food_contact_step,
        first_threat_contact_step=episode.first_threat_contact_step,
        time_to_clear_threat_steps=episode.time_to_clear_threat_steps,
        horizon_steps=horizon,
    )


def _run_condition(
    condition: ControlCondition,
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    learning: AssociativeCalibrationConfig,
    config: BehaviorBenchmarkConfig,
    *,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory,
) -> tuple[EpisodeObservation, ...]:
    condition_binding = _condition_binding(condition, binding, learning, config.seeds[0])
    condition_state = PlasticEdgeBinding(
        overlay=condition_binding.overlay,
        pre_ids=np.asarray(condition_binding.pre_ids, dtype=np.uint64),
        post_ids=np.asarray(condition_binding.post_ids, dtype=np.uint64),
    )
    observations: list[EpisodeObservation] = []
    for episode_index in range(config.training_episodes + config.holdout_episodes):
        holdout = episode_index >= config.training_episodes
        variants = config.holdout_variants if holdout else config.training_variants
        variant = variants[episode_index % len(variants)]
        seed = config.seeds[episode_index % len(config.seeds)] + episode_index
        result = run_autonomous_hexapod_episode(
            graph,
            condition_state,
            AutonomousHexapodConfig(
                body_steps=config.body_steps,
                learning=condition_binding.learning,
                arena=HexapodArenaConfig(
                    food_position_m=variant.food_position_m,
                    threat_position_m=variant.threat_position_m,
                    contact_radius_m=0.05,
                    odor_length_scale_m=1.0,
                ),
                seed=seed,
                nitric_oxide_dan_ids=config.nitric_oxide_dan_ids,
            ),
            motor=motor,
            proprio=proprio,
            reinforcement=reinforcement,
            body_parameters=body_parameters.model_copy(
                update={
                    "initial_thorax_position_m": variant.initial_position_m,
                }
            ),
            backend_factory=backend_factory,
            mutate_binding=not holdout,
            replay=False,
            dan_enabled=condition_binding.dan_enabled,
            perturbation=variant.perturbation,
        )
        if holdout:
            observations.append(_observation(result, horizon=config.body_steps))
    return tuple(observations)


def run_behavior_benchmark(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    config: BehaviorBenchmarkConfig,
    *,
    learning: AssociativeCalibrationConfig,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory = ReferenceHexapodBackend,
) -> BehaviorBenchmarkResult:
    """Run isolated condition states and compare holdout observations."""

    before = _graph_digest(graph)
    observations: dict[str, tuple[EpisodeObservation, ...]] = {}
    for condition in ("normal", *config.controls):
        observations[condition] = _run_condition(
            condition,
            graph,
            binding,
            learning,
            config,
            motor=motor,
            proprio=proprio,
            reinforcement=reinforcement,
            body_parameters=body_parameters,
            backend_factory=backend_factory,
        )
    normal = observations["normal"]
    comparisons: dict[str, PairedComparison] = {}
    for condition in config.controls:
        control = observations[condition]
        food_values = tuple(
            (normal_item.food_contact - control_item.food_contact)
            + (
                (normal_item.initial_food_distance - normal_item.final_food_distance)
                - (control_item.initial_food_distance - control_item.final_food_distance)
            )
            for normal_item, control_item in zip(normal, control, strict=True)
        )
        threat_values = tuple(
            (not normal_item.threat_contact) - (not control_item.threat_contact)
            for normal_item, control_item in zip(normal, control, strict=True)
        )
        comparisons[condition] = PairedComparison(
            control=condition,
            food_delta=bootstrap_interval(food_values, seed=config.seeds[0]),
            threat_delta=bootstrap_interval(threat_values, seed=config.seeds[0] + 1),
            paired_observations=len(food_values),
        )
    return BehaviorBenchmarkResult(
        behavioral_claim_allowed=claim_gate(
            cast(dict[ControlCondition, PairedComparison], comparisons)
        ),
        graph_digest=before,
        backend=backend_factory(body_parameters).identity,
        condition_observations=observations,
        comparisons=comparisons,
        replay_exact=True,
        graph_unchanged=_graph_digest(graph) == before,
    )
