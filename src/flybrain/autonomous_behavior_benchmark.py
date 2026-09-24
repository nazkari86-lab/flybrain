"""Bounded multi-episode behavioral benchmark over the autonomous hexapod loop."""

from __future__ import annotations

import hashlib
import math
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.associative_motor_loop import BackendFactory
from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    AutonomousHexapodResult,
    DescendingSpikeSummary,
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
    claim_gate,
    compare_conditions,
)
from flybrain.behavioral_perturbations import BehaviorVariant, world_digest
from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import BackendIdentity, ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap, PhaseEnvelopeAssumption
from flybrain.learning_memory import AutonomousLearningMemory
from flybrain.olfactory_interface import TaskOdorAssignment
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.retinal_interface import VisualInterfaceMap


class GeneralizationEvidence(BaseModel, frozen=True):
    """Structural proof that evaluation worlds are diverse and held out."""

    model_config = ConfigDict(extra="forbid")

    training_world_digests: tuple[str, ...]
    holdout_world_digests: tuple[str, ...]
    overlap_digests: tuple[str, ...]
    holdout_target_world_counts: dict[Literal["food", "threat"], int]
    unseen_worlds: bool
    sufficient_world_diversity: bool


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
    tactile_contact_input_ids: tuple[int, ...] = ()
    odor_a_input_ids: tuple[int, ...] = ()
    odor_b_input_ids: tuple[int, ...] = ()
    odor_a_left_input_ids: tuple[int, ...] = ()
    odor_a_right_input_ids: tuple[int, ...] = ()
    odor_b_left_input_ids: tuple[int, ...] = ()
    odor_b_right_input_ids: tuple[int, ...] = ()
    proprioceptive_encoding: Literal[
        "population_voltage", "source_equivalent_spikes"
    ] = "population_voltage"
    proprioceptive_spike_rate_hz: float = Field(default=150.0, gt=0.0)
    contact_radius_m: float = Field(default=0.05, gt=0.0)
    odor_length_scale_m: float = Field(default=1.0, gt=0.0)
    antenna_lateral_offset_m: float = Field(default=0.02, ge=0.0)
    visual_disc_radius_m: float = Field(default=0.05, gt=0.0)
    phase_envelope: PhaseEnvelopeAssumption = Field(default_factory=PhaseEnvelopeAssumption)
    neural_authority_nm: tuple[float, float, float] = (0.002, 0.0002, 0.0002)
    reinforcement_source: Literal[
        "contact_recruited", "contact_gated_neural_dan"
    ] = "contact_recruited"
    visual_interface: VisualInterfaceMap | None = None
    descending_map: DescendingMap | None = None
    odor_assignment: TaskOdorAssignment | None = None

    @model_validator(mode="after")
    def validate_benchmark(self) -> BehaviorBenchmarkConfig:
        if not all(math.isfinite(value) for value in (
            self.contact_radius_m,
            self.odor_length_scale_m,
            self.antenna_lateral_offset_m,
            self.visual_disc_radius_m,
        )):
            raise ValueError("benchmark arena dimensions must be finite")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("benchmark requires non-negative seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be unique independent replicates")
        if not self.training_variants or not self.holdout_variants:
            raise ValueError("benchmark requires training and holdout variants")
        if not any(
            variant.evaluation_target in {"food", "both"}
            for variant in self.holdout_variants
        ) or not any(
            variant.evaluation_target in {"threat", "both"}
            for variant in self.holdout_variants
        ):
            raise ValueError("benchmark requires both food and threat holdout targets")
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("benchmark controls must be unique")
        if any(condition not in CONDITIONS[1:] for condition in self.controls):
            raise ValueError("benchmark controls must be explicit non-normal conditions")
        if any(neuron_id <= 0 for neuron_id in self.tactile_contact_input_ids):
            raise ValueError("tactile-contact IDs must be positive")
        if len(set(self.tactile_contact_input_ids)) != len(self.tactile_contact_input_ids):
            raise ValueError("tactile-contact IDs must be unique")
        if bool(self.odor_a_input_ids) != bool(self.odor_b_input_ids):
            raise ValueError("benchmark explicit olfactory channels require both odors")
        side_channels = (
            self.odor_a_left_input_ids,
            self.odor_a_right_input_ids,
            self.odor_b_left_input_ids,
            self.odor_b_right_input_ids,
        )
        if any(not item for item in side_channels) and any(item for item in side_channels):
            raise ValueError("benchmark bilateral olfactory channels require all four sides")
        if any(side_channels) and (self.odor_a_input_ids or self.odor_b_input_ids):
            raise ValueError("benchmark olfactory channels cannot mix primary and bilateral inputs")
        all_channels = (
            self.odor_a_input_ids,
            self.odor_b_input_ids,
            *side_channels,
        )
        for ids in all_channels:
            if any(neuron_id <= 0 for neuron_id in ids):
                raise ValueError("benchmark olfactory IDs must be positive")
            if tuple(sorted(set(ids))) != ids:
                raise ValueError("benchmark olfactory IDs must be unique and sorted")
        nonempty = [set(ids) for ids in all_channels if ids]
        if sum(map(len, nonempty)) != len(set().union(*nonempty)):
            raise ValueError("benchmark explicit olfactory channels must be disjoint")
        return self

    def world_split(self) -> GeneralizationEvidence:
        """Measure world-level separation without trusting variant names."""

        training = tuple(sorted({world_digest(item) for item in self.training_variants}))
        holdout = tuple(sorted({world_digest(item) for item in self.holdout_variants}))
        overlap = tuple(sorted(set(training) & set(holdout)))
        targets: tuple[Literal["food", "threat"], ...] = ("food", "threat")
        counts: dict[Literal["food", "threat"], int] = {
            target: len({
                world_digest(item)
                for item in self.holdout_variants
                if item.evaluation_target in {target, "both"}
            })
            for target in targets
        }
        return GeneralizationEvidence(
            training_world_digests=training,
            holdout_world_digests=holdout,
            overlap_digests=overlap,
            holdout_target_world_counts=counts,
            unseen_worlds=not overlap,
            sufficient_world_diversity=all(value >= 2 for value in counts.values()),
        )


class TrainingSummary(BaseModel, frozen=True):
    """Training-only evidence that conditioning events actually occurred."""

    model_config = ConfigDict(extra="forbid")

    appetitive_contacts: int = Field(ge=0)
    aversive_contacts: int = Field(ge=0)
    dan_events: int = Field(ge=0)
    routed_dan_spike_events: int = Field(ge=0)


class NeuralActivitySummary(BaseModel, frozen=True):
    """Observed spikes along the learned path and final motor output."""

    model_config = ConfigDict(extra="forbid")

    plastic_kc_spikes: int = Field(ge=0)
    mbon_spikes: int = Field(ge=0)
    dan_spike_counts: dict[int, int]
    mbon_spike_counts: dict[int, int]
    mbon_mean_multipliers: dict[int, float]
    descending_spikes: DescendingSpikeSummary
    motor_spikes: int = Field(ge=0)
    routed_dan_spike_events: int = Field(ge=0)


class EpisodeEvidence(BaseModel, frozen=True):
    """Measured verification for every training and evaluation episode."""

    model_config = ConfigDict(extra="forbid")

    condition: ControlCondition
    replicate_seed: int
    episode_index: int
    holdout: bool
    plasticity_enabled: bool
    weights_unchanged: bool
    initial_memory_digest: str | None
    final_memory_digest: str
    memory_unchanged: bool
    replay_performed: bool
    replay_exact: bool
    graph_unchanged: bool
    final_support_count: int = Field(ge=0, le=6)
    final_fallen: bool
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BehaviorBenchmarkResult(BaseModel, frozen=True):
    """Auditable multi-condition observation and comparison artifact."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-behavior-benchmark-v1"] = (
        "autonomous-behavior-benchmark-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    behavioral_claim_allowed: bool
    odor_channel_model: Literal[
        "bipartite_registered_olfactory_assumption",
        "receptor_bank_olfactory_assumption",
        "measured_bilateral_receptor_channels",
    ]
    odor_assignment: TaskOdorAssignment | None
    proprioceptive_encoding: Literal[
        "population_voltage", "source_equivalent_spikes"
    ]
    proprioceptive_spike_rate_hz: float = Field(gt=0.0)
    phase_envelope: PhaseEnvelopeAssumption
    neural_authority_nm: tuple[float, float, float]
    reinforcement_source: Literal[
        "contact_recruited", "contact_gated_neural_dan"
    ]
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: BackendIdentity
    condition_observations: dict[str, tuple[EpisodeObservation, ...]]
    training_summaries: dict[str, TrainingSummary]
    holdout_neural_activity: dict[str, tuple[NeuralActivitySummary, ...]]
    comparisons: dict[str, PairedComparison]
    evidence_protocol: Literal["measured-replay-persistent-memory-v5"]
    world_split: GeneralizationEvidence
    generalization_verified: bool
    episode_evidence: tuple[EpisodeEvidence, ...]
    holdout_weights_frozen: bool
    holdout_memory_frozen: bool
    holdout_physical_stability_verified: bool
    unassisted_motor_output: bool
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


def _episode_seed(replicate_seed: int, episode_index: int) -> int:
    """Derive an episode stream independently of fixed intra-episode offsets."""

    sequence = np.random.SeedSequence([replicate_seed, episode_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _observation(
    result: AutonomousHexapodResult,
    *,
    horizon: int,
    evaluation_target: Literal["food", "threat", "both"],
) -> EpisodeObservation:
    episode = result
    return EpisodeObservation(
        evaluation_target=evaluation_target,
        food_contact=bool(episode.appetitive_contacts),
        threat_contact=bool(episode.aversive_contacts),
        initial_food_distance=episode.initial_food_distance,
        final_food_distance=episode.trace_distance_to_food[-1],
        initial_threat_distance=episode.initial_threat_distance,
        final_threat_distance=episode.trace_distance_to_threat[-1],
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
) -> tuple[
    tuple[EpisodeObservation, ...],
    TrainingSummary,
    tuple[NeuralActivitySummary, ...],
    tuple[EpisodeEvidence, ...],
]:
    observations: list[EpisodeObservation] = []
    holdout_neural_activity: list[NeuralActivitySummary] = []
    evidence: list[EpisodeEvidence] = []
    appetitive_contacts = 0
    aversive_contacts = 0
    dan_events = 0
    routed_dan_spike_events = 0
    for replicate_seed in config.seeds:
        memory: AutonomousLearningMemory | None = None
        condition_binding = _condition_binding(
            condition, binding, learning, replicate_seed
        )
        condition_state = PlasticEdgeBinding(
            overlay=condition_binding.overlay,
            pre_ids=np.asarray(condition_binding.pre_ids, dtype=np.uint64),
            post_ids=np.asarray(condition_binding.post_ids, dtype=np.uint64),
        )
        episode_plan = tuple(
            (False, variant)
            for _ in range(config.training_episodes)
            for variant in config.training_variants
        ) + tuple(
            (True, variant)
            for _ in range(config.holdout_episodes)
            for variant in config.holdout_variants
        )
        for episode_index, (holdout, variant) in enumerate(episode_plan):
            episode_seed = _episode_seed(replicate_seed, episode_index)
            initial_multipliers = tuple(float(x) for x in condition_state.overlay.multipliers)
            plasticity_enabled = condition_binding.plasticity_enabled and not holdout
            initial_memory_digest = memory.digest if memory is not None else None
            result = run_autonomous_hexapod_episode(
                graph,
                condition_state,
                AutonomousHexapodConfig(
                    body_steps=config.body_steps,
                    learning=condition_binding.learning,
                    arena=HexapodArenaConfig(
                        food_position_m=variant.food_position_m,
                        threat_position_m=variant.threat_position_m,
                        contact_radius_m=config.contact_radius_m,
                        odor_length_scale_m=config.odor_length_scale_m,
                        antenna_lateral_offset_m=config.antenna_lateral_offset_m,
                        visual_disc_radius_m=config.visual_disc_radius_m,
                    ),
                    seed=episode_seed,
                    nitric_oxide_dan_ids=config.nitric_oxide_dan_ids,
                    tactile_contact_input_ids=config.tactile_contact_input_ids,
                    odor_a_input_ids=config.odor_a_input_ids,
                    odor_b_input_ids=config.odor_b_input_ids,
                    odor_a_left_input_ids=config.odor_a_left_input_ids,
                    odor_a_right_input_ids=config.odor_a_right_input_ids,
                    odor_b_left_input_ids=config.odor_b_left_input_ids,
                    odor_b_right_input_ids=config.odor_b_right_input_ids,
                    proprioceptive_encoding=config.proprioceptive_encoding,
                    proprioceptive_spike_rate_hz=config.proprioceptive_spike_rate_hz,
                    phase_envelope=config.phase_envelope,
                    neural_authority_nm=config.neural_authority_nm,
                    reinforcement_source=config.reinforcement_source,
                    visual_interface=config.visual_interface,
                    descending_map=config.descending_map,
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
                replay=True,
                dan_enabled=condition_binding.dan_enabled,
                plasticity_enabled=plasticity_enabled,
                perturbation=variant.perturbation,
                learning_memory=memory,
            )
            evidence.append(EpisodeEvidence(
                condition=condition,
                replicate_seed=replicate_seed,
                episode_index=episode_index,
                holdout=holdout,
                plasticity_enabled=plasticity_enabled,
                weights_unchanged=(initial_multipliers == result.final_multipliers),
                initial_memory_digest=initial_memory_digest,
                final_memory_digest=result.learning_memory.digest,
                memory_unchanged=(memory == result.learning_memory),
                replay_performed=result.replay_performed,
                replay_exact=result.replay_exact,
                graph_unchanged=result.graph_unchanged,
                final_support_count=result.final_body.support_count,
                final_fallen=result.final_body.fallen,
                trace_digest=result.trace_digest,
            ))
            if holdout:
                observations.append(
                    _observation(
                        result,
                        horizon=config.body_steps,
                        evaluation_target=variant.evaluation_target,
                    )
                )
                holdout_neural_activity.append(
                    NeuralActivitySummary(
                        plastic_kc_spikes=result.plastic_kc_spikes,
                        mbon_spikes=result.mbon_spikes,
                        dan_spike_counts=result.dan_spike_counts,
                        mbon_spike_counts=result.mbon_spike_counts,
                        mbon_mean_multipliers=result.mbon_mean_multipliers,
                        descending_spikes=result.descending_spikes,
                        motor_spikes=result.motor_spikes,
                        routed_dan_spike_events=result.routed_dan_spike_events,
                    )
                )
            else:
                memory = result.learning_memory
                appetitive_contacts += result.appetitive_contacts
                aversive_contacts += result.aversive_contacts
                dan_events += result.dan_events
                routed_dan_spike_events += result.routed_dan_spike_events
    return (
        tuple(observations),
        TrainingSummary(
            appetitive_contacts=appetitive_contacts,
            aversive_contacts=aversive_contacts,
            dan_events=dan_events,
            routed_dan_spike_events=routed_dan_spike_events,
        ),
        tuple(holdout_neural_activity),
        tuple(evidence),
    )


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
    training_summaries: dict[str, TrainingSummary] = {}
    holdout_neural_activity: dict[str, tuple[NeuralActivitySummary, ...]] = {}
    evidence: list[EpisodeEvidence] = []
    for condition in ("normal", *config.controls):
        (condition_observations, training_summary,
         condition_neural_activity, condition_evidence) = _run_condition(
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
        observations[condition] = condition_observations
        training_summaries[condition] = training_summary
        holdout_neural_activity[condition] = condition_neural_activity
        evidence.extend(condition_evidence)
    normal = observations["normal"]
    comparisons: dict[str, PairedComparison] = {}
    for condition in config.controls:
        comparisons[condition] = compare_conditions(
            normal,
            observations[condition],
            control=condition,
            seed=config.seeds[0],
            food_replicate_size=(
                config.holdout_episodes
                * sum(
                    variant.evaluation_target in {"food", "both"}
                    for variant in config.holdout_variants
                )
            ),
            threat_replicate_size=(
                config.holdout_episodes
                * sum(
                    variant.evaluation_target in {"threat", "both"}
                    for variant in config.holdout_variants
                )
            ),
        )
    graph_unchanged = _graph_digest(graph) == before and all(
        item.graph_unchanged for item in evidence
    )
    replay_exact = bool(evidence) and all(
        item.replay_performed and item.replay_exact for item in evidence
    )
    holdout_weights_frozen = all(
        not item.plasticity_enabled and item.weights_unchanged
        for item in evidence if item.holdout
    )
    holdout_memory_frozen = all(
        item.memory_unchanged for item in evidence if item.holdout
    )
    normal_holdouts = tuple(
        item for item in evidence if item.condition == "normal" and item.holdout
    )
    holdout_physical_stability_verified = bool(normal_holdouts) and all(
        not item.final_fallen and item.final_support_count >= 3
        for item in normal_holdouts
    )
    world_split = config.world_split()
    generalization_verified = (
        world_split.unseen_worlds and world_split.sufficient_world_diversity
    )
    return BehaviorBenchmarkResult(
        behavioral_claim_allowed=(
            config.odor_assignment is not None
            and holdout_weights_frozen and holdout_memory_frozen
            and holdout_physical_stability_verified
            and config.reinforcement_source == "contact_gated_neural_dan"
            and config.phase_envelope.amplitude_nm == 0.0
        ) and claim_gate(
            cast(dict[ControlCondition, PairedComparison], comparisons),
            replay_exact=replay_exact,
            graph_unchanged=graph_unchanged,
            generalization_verified=generalization_verified,
        ),
        odor_channel_model=(
            "measured_bilateral_receptor_channels"
            if config.odor_a_left_input_ids
            else "receptor_bank_olfactory_assumption"
            if config.odor_a_input_ids
            else "bipartite_registered_olfactory_assumption"
        ),
        odor_assignment=config.odor_assignment,
        proprioceptive_encoding=config.proprioceptive_encoding,
        proprioceptive_spike_rate_hz=config.proprioceptive_spike_rate_hz,
        phase_envelope=config.phase_envelope,
        neural_authority_nm=config.neural_authority_nm,
        reinforcement_source=config.reinforcement_source,
        graph_digest=before,
        backend=backend_factory(body_parameters).identity,
        condition_observations=observations,
        training_summaries=training_summaries,
        holdout_neural_activity=holdout_neural_activity,
        comparisons=comparisons,
        evidence_protocol="measured-replay-persistent-memory-v5",
        world_split=world_split,
        generalization_verified=generalization_verified,
        episode_evidence=tuple(evidence),
        holdout_weights_frozen=holdout_weights_frozen,
        holdout_memory_frozen=holdout_memory_frozen,
        holdout_physical_stability_verified=holdout_physical_stability_verified,
        unassisted_motor_output=config.phase_envelope.amplitude_nm == 0.0,
        replay_exact=replay_exact,
        graph_unchanged=graph_unchanged,
    )
