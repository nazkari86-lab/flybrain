"""Contact-driven associative learning on the complete hexapod motor interface."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.associative_motor_loop import BackendFactory
from flybrain.autonomous_learning_benchmark import (
    AssociativeCalibrationConfig,
)
from flybrain.behavioral_perturbations import BodyPerturbation, apply_perturbation
from flybrain.descending_interface import DescendingMap
from flybrain.embodied_world import FlyBody
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import BackendIdentity, ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodBody, HexapodParameters, HexapodTorque
from flybrain.hexapod_motor import (
    CANONICAL_MOTOR_GROUPS,
    HexapodMotorDecoder,
    HexapodMotorMap,
    PhaseEnvelopeAssumption,
)
from flybrain.learning_memory import AutonomousLearningMemory
from flybrain.mushroom_body_learning import (
    MushroomBodyLearning,
    derive_approach_mbon_ids,
)
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.proprioceptive_interface import (
    ProprioceptiveCalibration,
    ProprioceptiveEncoder,
    ProprioceptiveMap,
    observe_proprioception,
)
from flybrain.reinforcement_interface import AnonymousContact, ReinforcementInterface
from flybrain.retinal_interface import (
    VisualDisc,
    VisualInterfaceEncoder,
    VisualInterfaceMap,
    observe_retina,
    rebase_overlapping_history,
)
from flybrain.shiu import ShiuState, poisson_voltage_events, simulate_shiu
from flybrain.slow_memory import SlowMemoryParameters, SlowMemoryState


class HexapodArenaConfig(BaseModel, frozen=True):
    """Physical source locations kept inside the environment boundary."""

    model_config = ConfigDict(extra="forbid")

    food_position_m: tuple[float, float]
    threat_position_m: tuple[float, float]
    contact_radius_m: float = Field(gt=0.0)
    odor_length_scale_m: float = Field(gt=0.0)
    antenna_lateral_offset_m: float = Field(default=0.02, ge=0.0)
    visual_disc_radius_m: float = Field(default=0.05, gt=0.0)
    odor_plume_depth: float = Field(default=0.25, ge=0.0, lt=1.0)
    odor_plume_frequency_hz: float = Field(default=4.0, gt=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        values = (
            *self.food_position_m,
            *self.threat_position_m,
            self.contact_radius_m,
            self.odor_length_scale_m,
            self.antenna_lateral_offset_m,
            self.visual_disc_radius_m,
            self.odor_plume_depth,
            self.odor_plume_frequency_hz,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("hexapod arena values must be finite")
        return self


class AutonomousHexapodConfig(BaseModel, frozen=True):
    """Bounded autonomous episode inputs without schedules or desired actions."""

    model_config = ConfigDict(extra="forbid")

    body_steps: int = Field(gt=0)
    learning: AssociativeCalibrationConfig
    arena: HexapodArenaConfig
    seed: int = Field(ge=0)
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
    phase_envelope: PhaseEnvelopeAssumption = Field(default_factory=PhaseEnvelopeAssumption)
    neural_authority_nm: tuple[float, float, float] = (0.002, 0.0002, 0.0002)
    reinforcement_source: Literal[
        "contact_recruited", "contact_gated_neural_dan"
    ] = "contact_recruited"
    visual_interface: VisualInterfaceMap | None = None
    descending_map: DescendingMap | None = None
    slow_memory_parameters: SlowMemoryParameters = Field(
        default_factory=SlowMemoryParameters
    )

    @model_validator(mode="after")
    def validate_sensory_path(self) -> Self:
        if self.learning.input_mode != "sensory_path":
            raise ValueError("autonomous hexapod requires the canonical sensory path")
        if not self.learning.sensory_input_ids:
            raise ValueError("autonomous hexapod requires declared sensory input IDs")
        if any(neuron_id <= 0 for neuron_id in self.nitric_oxide_dan_ids):
            raise ValueError("nitric-oxide DAN IDs must be positive")
        if len(set(self.nitric_oxide_dan_ids)) != len(self.nitric_oxide_dan_ids):
            raise ValueError("nitric-oxide DAN IDs must be unique")
        if any(neuron_id <= 0 for neuron_id in self.tactile_contact_input_ids):
            raise ValueError("tactile-contact IDs must be positive")
        if len(set(self.tactile_contact_input_ids)) != len(self.tactile_contact_input_ids):
            raise ValueError("tactile-contact IDs must be unique")
        if bool(self.odor_a_input_ids) != bool(self.odor_b_input_ids):
            raise ValueError("explicit olfactory channels must declare both odor A and odor B")
        side_channels = (
            self.odor_a_left_input_ids,
            self.odor_a_right_input_ids,
            self.odor_b_left_input_ids,
            self.odor_b_right_input_ids,
        )
        if any(not item for item in side_channels) and any(item for item in side_channels):
            raise ValueError("explicit bilateral olfactory channels require all four sides")
        if any(side_channels) and (self.odor_a_input_ids or self.odor_b_input_ids):
            raise ValueError("olfactory channels cannot mix primary and bilateral inputs")
        named_channels = (
            ("odor A", self.odor_a_input_ids),
            ("odor B", self.odor_b_input_ids),
            ("odor A left", self.odor_a_left_input_ids),
            ("odor A right", self.odor_a_right_input_ids),
            ("odor B left", self.odor_b_left_input_ids),
            ("odor B right", self.odor_b_right_input_ids),
        )
        for name, ids in named_channels:
            if any(neuron_id <= 0 for neuron_id in ids):
                raise ValueError(f"{name} IDs must be positive")
            if tuple(sorted(set(ids))) != ids:
                raise ValueError(f"{name} IDs must be unique and sorted")
        nonempty = [set(ids) for _, ids in named_channels if ids]
        if sum(map(len, nonempty)) != len(set().union(*nonempty)):
            raise ValueError("explicit olfactory channels must be disjoint")
        declared_sensory = set(self.learning.sensory_input_ids)
        declared_odor = set().union(*(set(ids) for _, ids in named_channels))
        if declared_odor - declared_sensory:
            raise ValueError("explicit olfactory channel IDs must be declared sensory inputs")
        if not math.isfinite(self.proprioceptive_spike_rate_hz):
            raise ValueError("proprioceptive spike rate must be finite")
        self.slow_memory_parameters.validate()
        return self


class DescendingSpikeSummary(BaseModel, frozen=True):
    """Observed spikes in registered descending populations, without decoding commands."""

    model_config = ConfigDict(extra="forbid")

    d_na02_left: int = Field(ge=0)
    d_na02_right: int = Field(ge=0)
    d_ng13_left: int = Field(ge=0)
    d_ng13_right: int = Field(ge=0)
    mdn_left: int = Field(ge=0)
    mdn_right: int = Field(ge=0)


def _approach_mbon_ids(
    learning: AssociativeCalibrationConfig,
    reinforcement: ReinforcementInterface,
) -> np.ndarray:
    if learning.learning_parameters.plasticity_mode != "valence_compartmental":
        return np.empty(0, dtype=np.uint64)
    return derive_approach_mbon_ids(
        dan_ids=np.asarray(
            [dan_id for dan_id, _ in learning.dan_to_mbon_pairs], dtype=np.uint64
        ),
        dan_post_ids=np.asarray(
            [post_id for _, post_id in learning.dan_to_mbon_pairs], dtype=np.uint64
        ),
        appetitive_dan_ids=np.asarray(
            reinforcement.appetitive_dan_ids, dtype=np.uint64
        ),
        aversive_dan_ids=np.asarray(reinforcement.aversive_dan_ids, dtype=np.uint64),
    )


class AutonomousHexapodResult(BaseModel, frozen=True):
    """Auditable closed-loop trace summary with conservative claim gating."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-hexapod-contact-learning-v1"] = (
        "autonomous-hexapod-contact-learning-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    odor_channel_model: Literal[
        "bipartite_registered_olfactory_assumption",
        "receptor_bank_olfactory_assumption",
        "measured_bilateral_receptor_channels",
    ] = "bipartite_registered_olfactory_assumption"
    proprioceptive_encoding: Literal[
        "population_voltage", "source_equivalent_spikes"
    ] = "population_voltage"
    proprioceptive_spike_rate_hz: float = Field(default=150.0, gt=0.0)
    phase_envelope: PhaseEnvelopeAssumption
    neural_authority_nm: tuple[float, float, float]
    plasticity_mode: Literal["depression_only", "valence_compartmental"]
    valence_compartment_model: Literal["none", "dan_route_majority_assumption"]
    approach_mbon_count: int = Field(ge=0)
    tactile_contact_model: Literal[
        "none", "uniform_registered_vnc_tactile_assumption"
    ]
    autonomous_behavior_claim_allowed: Literal[False] = False
    backend: BackendIdentity
    replay_performed: bool
    replay_exact: bool
    graph_unchanged: bool
    body_steps: int = Field(gt=0)
    neural_steps: int = Field(gt=0)
    appetitive_contacts: int = Field(ge=0)
    aversive_contacts: int = Field(ge=0)
    reinforcement_source: Literal[
        "contact_recruited", "contact_gated_neural_dan"
    ]
    dan_events: int = Field(ge=0)
    routed_dan_spike_events: int = Field(ge=0)
    dan_spike_counts: dict[int, int]
    plastic_kc_spikes: int = Field(ge=0)
    mbon_spikes: int = Field(ge=0)
    mbon_spike_counts: dict[int, int]
    mbon_mean_multipliers: dict[int, float]
    descending_spikes: DescendingSpikeSummary
    motor_spikes: int = Field(ge=0)
    active_motor_groups: int = Field(ge=0, le=len(CANONICAL_MOTOR_GROUPS))
    all_motor_groups_active: bool
    proprioceptive_events: int = Field(ge=0)
    tactile_contact_events: int = Field(ge=0)
    visual_source_events: int = Field(ge=0)
    slow_memory_enabled: bool
    slow_memory_edges: int = Field(ge=0)
    slow_memory_dopamine_effect_max: float = Field(ge=0.0, le=1.0)
    slow_memory_nitric_oxide_effect_max: float = Field(ge=0.0, le=1.0)
    final_multipliers: tuple[float, ...]
    learning_memory: AutonomousLearningMemory
    initial_food_distance: float = Field(ge=0.0)
    initial_threat_distance: float = Field(ge=0.0)
    trace_distance_to_food: tuple[float, ...]
    trace_distance_to_threat: tuple[float, ...]
    first_food_contact_step: int | None
    first_threat_contact_step: int | None
    time_to_clear_threat_steps: int | None
    final_body: HexapodBody
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _Trace:
    backend: BackendIdentity
    appetitive_contacts: int
    aversive_contacts: int
    reinforcement_source: Literal[
        "contact_recruited", "contact_gated_neural_dan"
    ]
    dan_events: int
    routed_dan_spike_events: int
    dan_spike_counts: dict[int, int]
    plastic_kc_spikes: int
    mbon_spikes: int
    mbon_spike_counts: dict[int, int]
    mbon_mean_multipliers: dict[int, float]
    descending_spikes: DescendingSpikeSummary
    motor_spikes: int
    active_motor_groups: tuple[str, ...]
    proprioceptive_events: int
    tactile_contact_events: int
    visual_source_events: int
    slow_memory_enabled: bool
    slow_memory_edges: int
    slow_memory_dopamine_effect_max: float
    slow_memory_nitric_oxide_effect_max: float
    final_multipliers: tuple[float, ...]
    learning_memory: AutonomousLearningMemory
    initial_food_distance: float
    initial_threat_distance: float
    trace_distance_to_food: tuple[float, ...]
    trace_distance_to_threat: tuple[float, ...]
    first_food_contact_step: int | None
    first_threat_contact_step: int | None
    time_to_clear_threat_steps: int | None
    final_body: HexapodBody
    body_digest: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _clone_binding(binding: PlasticEdgeBinding) -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=binding.overlay.copy(),
        pre_ids=binding.pre_ids.copy(),
        post_ids=binding.post_ids.copy(),
    )


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


def _distance(body: HexapodBody, point: tuple[float, float]) -> float:
    return math.hypot(
        body.thorax_position_m[0] - point[0],
        body.thorax_position_m[1] - point[1],
    )


def _odor_intensities(
    body: HexapodBody,
    arena: HexapodArenaConfig,
    time_s: float = 0.0,
) -> tuple[float, float]:
    """Return anonymous concentration channels with a deterministic moving plume."""

    def concentration(point: tuple[float, float], phase: float) -> float:
        distance = _distance(body, point)
        base = math.exp(-distance / arena.odor_length_scale_m)
        modulation = 1.0 + arena.odor_plume_depth * math.sin(
            2.0 * math.pi * arena.odor_plume_frequency_hz * time_s + phase
        )
        return max(0.0, min(1.0, base * modulation))

    return (
        concentration(arena.food_position_m, 0.0),
        concentration(arena.threat_position_m, math.pi / 2.0),
    )


def _bilateral_odor_intensities(
    body: HexapodBody, arena: HexapodArenaConfig, time_s: float = 0.0
) -> tuple[float, float, float, float]:
    """Approximate paired antennal samples from body-relative source geometry."""

    def sample(point: tuple[float, float], lateral: float, source_phase: float) -> float:
        offset = (
            -math.sin(body.thorax_yaw_rad) * lateral,
            math.cos(body.thorax_yaw_rad) * lateral,
        )
        sensor = (
            body.thorax_position_m[0] + offset[0],
            body.thorax_position_m[1] + offset[1],
        )
        distance = math.hypot(sensor[0] - point[0], sensor[1] - point[1])
        base = math.exp(-distance / arena.odor_length_scale_m)
        phase = (
            2.0 * math.pi * arena.odor_plume_frequency_hz * time_s + source_phase
        )
        return max(0.0, min(1.0, base * (1.0 + arena.odor_plume_depth * math.sin(phase))))

    return (
        sample(arena.food_position_m, arena.antenna_lateral_offset_m, 0.0),
        sample(arena.food_position_m, -arena.antenna_lateral_offset_m, 0.0),
        sample(arena.threat_position_m, arena.antenna_lateral_offset_m, math.pi / 2.0),
        sample(arena.threat_position_m, -arena.antenna_lateral_offset_m, math.pi / 2.0),
    )


def _retinal_body(body: HexapodBody) -> FlyBody:
    return FlyBody(
        x=body.thorax_position_m[0],
        y=body.thorax_position_m[1],
        heading_rad=body.thorax_yaw_rad,
        forward_speed=math.hypot(*body.thorax_velocity_m_s),
        angular_speed=body.thorax_yaw_rate_rad_s,
        energy=body.energy_j,
        leg_contacts=tuple(leg.contact for leg in body.legs),
    )


def _anonymous_visual_scene(
    body: HexapodBody,
    arena: HexapodArenaConfig,
) -> tuple[VisualDisc, ...]:
    """Project anonymous arena geometry without passing object labels inward."""

    points = (arena.food_position_m, arena.threat_position_m)
    discs: list[VisualDisc] = []
    for index, (x, y) in enumerate(points):
        dx = x - body.thorax_position_m[0]
        dy = y - body.thorax_position_m[1]
        distance = math.hypot(dx, dy)
        radius = arena.visual_disc_radius_m
        if distance <= radius:
            angle = math.atan2(dy, dx) if distance > 1e-9 else index * math.pi
            distance = radius * 2.0
            x = body.thorax_position_m[0] + distance * math.cos(angle)
            y = body.thorax_position_m[1] + distance * math.sin(angle)
            radius *= 0.25
        discs.append(VisualDisc(x=x, y=y, radius=radius))
    return tuple(discs)


def _run(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    config: AutonomousHexapodConfig,
    *,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory,
    proprioceptive_calibration: ProprioceptiveCalibration,
    perturbation: BodyPerturbation,
    dan_enabled: bool,
    plasticity_enabled: bool,
    learning_memory: AutonomousLearningMemory | None,
    memory_context: str,
) -> _Trace:
    motor.validate_graph(graph)
    proprio.validate_graph(graph)
    learning = config.learning
    approach_mbon_ids = _approach_mbon_ids(learning, reinforcement)
    neural_window_s = (
        learning.neural_chunk_steps * learning.shiu_parameters.dt_ms / 1000.0
    )
    if not math.isclose(neural_window_s, body_parameters.dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("neural chunk duration must equal the backend timestep")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    dan_ids = {
        int(dan_id)
        for dan_id, _ in learning.dan_to_mbon_pairs
    }
    required = {
        *learning.sensory_input_ids,
        *learning.cue_ids,
        *learning.mbon_ids,
        *(pre for pre, _ in learning.dan_to_mbon_pairs),
        *(post for _, post in learning.dan_to_mbon_pairs),
        *binding.pre_ids.tolist(),
        *binding.post_ids.tolist(),
        *config.tactile_contact_input_ids,
        *config.odor_a_input_ids,
        *config.odor_b_input_ids,
        *config.odor_a_left_input_ids,
        *config.odor_a_right_input_ids,
        *config.odor_b_left_input_ids,
        *config.odor_b_right_input_ids,
    }
    if config.visual_interface is not None:
        required.update(
            value
            for group in (
                config.visual_interface.left_r1_r6_ids,
                config.visual_interface.right_r1_r6_ids,
                config.visual_interface.left_hs_ids,
                config.visual_interface.right_hs_ids,
                config.visual_interface.left_lc16_ids,
                config.visual_interface.right_lc16_ids,
            )
            for value in group
        )
    if config.descending_map is not None:
        required.update(
            neuron_id
            for ids in config.descending_map.named_populations().values()
            for neuron_id in ids
        )
    missing = sorted(required - set(index_by_id))
    if missing:
        raise ValueError(f"autonomous hexapod IDs absent from graph: {missing}")
    silenced = np.zeros(graph.neuron_count, dtype=np.bool_)
    if not dan_enabled:
        silenced[[index_by_id[neuron_id] for neuron_id in dan_ids]] = True
    state = ShiuState.initial(
        graph.neuron_count,
        params=learning.shiu_parameters,
        seed=config.seed,
    )
    learner = MushroomBodyLearning(
        overlay=(
            binding.overlay.copy()
            if config.nitric_oxide_dan_ids
            else binding.overlay
        ),
        edge_pre_ids=binding.pre_ids,
        edge_post_ids=binding.post_ids,
        dan_ids=np.asarray([item[0] for item in learning.dan_to_mbon_pairs], dtype=np.uint64),
        dan_post_ids=np.asarray(
            [item[1] for item in learning.dan_to_mbon_pairs], dtype=np.uint64
        ),
        parameters=learning.learning_parameters,
        appetitive_dan_ids=np.asarray(
            reinforcement.appetitive_dan_ids, dtype=np.uint64
        ),
        aversive_dan_ids=np.asarray(reinforcement.aversive_dan_ids, dtype=np.uint64),
        approach_mbon_ids=approach_mbon_ids,
    )
    slow_state: SlowMemoryState | None = None
    if config.nitric_oxide_dan_ids:
        no_dan_ids = set(config.nitric_oxide_dan_ids)
        no_target_ids = np.asarray(
            [
                post_id
                for dan_id, post_id in learning.dan_to_mbon_pairs
                if dan_id in no_dan_ids
            ],
            dtype=np.uint64,
        )
        competent = np.isin(binding.post_ids, no_target_ids)
        if not np.any(competent):
            raise ValueError(
                "nitric-oxide DANs do not route to any bound KC-to-MBON edge"
            )
        slow_state = SlowMemoryState.initial(nitric_oxide_competent=competent)
    if learning_memory is not None:
        learning_memory.restore(learner, slow_state, binding.overlay.multipliers, memory_context)
    backend = backend_factory(body_parameters)
    initial_body = backend.observe()
    initial_food_distance = _distance(initial_body, config.arena.food_position_m)
    initial_threat_distance = _distance(initial_body, config.arena.threat_position_m)
    decoder = HexapodMotorDecoder(
        motor,
        neural_authority_nm=config.neural_authority_nm,
        phase_envelope=config.phase_envelope,
        descending_map=config.descending_map,
    )
    proprio_encoder = ProprioceptiveEncoder(proprio, proprioceptive_calibration)
    source_indices = np.asarray(
        [index_by_id[item] for item in learning.sensory_input_ids], dtype=np.int64
    )
    if config.odor_a_left_input_ids:
        food_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_a_left_input_ids], dtype=np.int64
        )
        food_right_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_a_right_input_ids], dtype=np.int64
        )
        threat_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_b_left_input_ids], dtype=np.int64
        )
        threat_right_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_b_right_input_ids], dtype=np.int64
        )
    elif config.odor_a_input_ids:
        food_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_a_input_ids], dtype=np.int64
        )
        threat_source_indices = np.asarray(
            [index_by_id[item] for item in config.odor_b_input_ids], dtype=np.int64
        )
        food_right_source_indices = threat_right_source_indices = np.empty(0, dtype=np.int64)
    else:
        food_source_indices = source_indices[::2]
        threat_source_indices = source_indices[1::2]
        food_right_source_indices = threat_right_source_indices = np.empty(0, dtype=np.int64)
    visual_encoder = (
        VisualInterfaceEncoder(config.visual_interface)
        if config.visual_interface is not None
        else None
    )
    previous_visual_scene: tuple[VisualDisc, ...] | None = None
    tactile_source_indices = np.asarray(
        [index_by_id[item] for item in config.tactile_contact_input_ids], dtype=np.int64
    )
    plastic_kc_ids = set(int(neuron_id) for neuron_id in binding.pre_ids)
    mbon_ids = set(learning.mbon_ids)
    mbon_spike_counts = {neuron_id: 0 for neuron_id in sorted(mbon_ids)}
    descending_ids = (
        {
            name: set(ids)
            for name, ids in config.descending_map.named_populations().items()
        }
        if config.descending_map is not None
        else {}
    )
    descending_counts = {name: 0 for name in descending_ids}
    motor_owner = {
        neuron_id: group.name for group in motor.groups for neuron_id in group.neuron_ids
    }
    motor_spikes = 0
    active_groups: set[str] = set()
    proprioceptive_events = 0
    tactile_contact_events = 0
    visual_source_events = 0
    appetitive_contacts = 0
    aversive_contacts = 0
    dan_events = 0
    routed_dan_spike_events = 0
    dan_spike_counts = {neuron_id: 0 for neuron_id in sorted(dan_ids)}
    plastic_kc_spikes = 0
    mbon_spikes = 0
    body_trace = []
    distance_to_food: list[float] = []
    distance_to_threat: list[float] = []
    first_food_contact_step: int | None = None
    first_threat_contact_step: int | None = None
    time_to_clear_threat_steps: int | None = None
    torque_queue: list[HexapodTorque] = []
    for step_index in range(config.body_steps):
        body = backend.observe()
        time_s = step_index * body_parameters.dt_s
        food_intensity, threat_intensity = _odor_intensities(body, config.arena, time_s)
        if food_right_source_indices.size:
            food_left, food_right, threat_left, threat_right = _bilateral_odor_intensities(
                body, config.arena, time_s
            )
            food_intensity, threat_intensity = food_left, threat_left
        food_sensory_params = replace(
            learning.shiu_parameters,
            poisson_rate_hz=learning.shiu_parameters.poisson_rate_hz * food_intensity,
        )
        threat_sensory_params = replace(
            learning.shiu_parameters,
            poisson_rate_hz=learning.shiu_parameters.poisson_rate_hz * threat_intensity,
        )
        scheduled: dict[int, list[tuple[int, float]]] = {}
        if visual_encoder is not None:
            current_visual_scene = _anonymous_visual_scene(body, config.arena)
            if previous_visual_scene is None:
                previous_visual_scene = current_visual_scene
            previous_visual_scene = rebase_overlapping_history(
                _retinal_body(body), previous_visual_scene, current_visual_scene
            )
            observation = observe_retina(
                _retinal_body(body), previous_visual_scene, current_visual_scene
            )
            visual_events = visual_encoder.encode_photoreceptor_spikes(
                observation,
                steps=learning.neural_chunk_steps,
                seed=config.seed + state.step + 3_000_009,
                dt_ms=learning.shiu_parameters.dt_ms,
            )
            visual_source_events += len(visual_events)
            for event in visual_events:
                scheduled.setdefault(state.step + event.step, []).extend(
                    zip(
                        (index_by_id[neuron_id] for neuron_id in event.neuron_ids),
                        event.voltages,
                        strict=True,
                    )
                )
            previous_visual_scene = current_visual_scene
        for relative_step, (indices, voltages) in poisson_voltage_events(
            food_source_indices,
            steps=learning.neural_chunk_steps,
            params=food_sensory_params,
            seed=config.seed + state.step,
        ).items():
            scheduled.setdefault(state.step + relative_step, []).extend(
                zip(indices.tolist(), voltages.tolist(), strict=True)
            )
        for relative_step, (indices, voltages) in poisson_voltage_events(
            threat_source_indices,
            steps=learning.neural_chunk_steps,
            params=threat_sensory_params,
            seed=config.seed + state.step + 1_000_003,
        ).items():
            scheduled.setdefault(state.step + relative_step, []).extend(
                zip(indices.tolist(), voltages.tolist(), strict=True)
            )
        if food_right_source_indices.size:
            for relative_step, (indices, voltages) in poisson_voltage_events(
                food_right_source_indices,
                steps=learning.neural_chunk_steps,
                params=replace(
                    learning.shiu_parameters,
                    poisson_rate_hz=learning.shiu_parameters.poisson_rate_hz * food_right,
                ),
                seed=config.seed + state.step + 3_000_017,
            ).items():
                scheduled.setdefault(state.step + relative_step, []).extend(
                    zip(indices.tolist(), voltages.tolist(), strict=True)
                )
            for relative_step, (indices, voltages) in poisson_voltage_events(
                threat_right_source_indices,
                steps=learning.neural_chunk_steps,
                params=replace(
                    learning.shiu_parameters,
                    poisson_rate_hz=learning.shiu_parameters.poisson_rate_hz * threat_right,
                ),
                seed=config.seed + state.step + 4_000_019,
            ).items():
                scheduled.setdefault(state.step + relative_step, []).extend(
                    zip(indices.tolist(), voltages.tolist(), strict=True)
                )
        physical_contact = (
            _distance(body, config.arena.food_position_m) <= config.arena.contact_radius_m
            or _distance(body, config.arena.threat_position_m) <= config.arena.contact_radius_m
        )
        if physical_contact and tactile_source_indices.size:
            tactile_events = poisson_voltage_events(
                tactile_source_indices,
                steps=learning.neural_chunk_steps,
                params=learning.shiu_parameters,
                seed=config.seed + state.step + 2_000_006,
            )
            tactile_contact_events += sum(
                len(indices) for indices, _ in tactile_events.values()
            )
            for relative_step, (indices, voltages) in tactile_events.items():
                scheduled.setdefault(state.step + relative_step, []).extend(
                    zip(indices.tolist(), voltages.tolist(), strict=True)
                )
        proprio_observation = observe_proprioception(
            body, body_parameters, proprioceptive_calibration
        )
        if config.proprioceptive_encoding == "source_equivalent_spikes":
            proprio_events = proprio_encoder.encode_source_equivalent_spikes(
                proprio_observation,
                steps=learning.neural_chunk_steps,
                seed=config.seed + state.step + 4_000_013,
                rate_hz=config.proprioceptive_spike_rate_hz,
                dt_ms=learning.shiu_parameters.dt_ms,
            )
        else:
            proprio_events = proprio_encoder.encode(
                proprio_observation,
                step=state.step,
            )
        proprioceptive_events += len(proprio_events)
        for event in proprio_events:
            event_step = (
                state.step + event.step
                if config.proprioceptive_encoding == "source_equivalent_spikes"
                else event.step
            )
            scheduled.setdefault(event_step, []).extend(
                zip(
                    (index_by_id[item] for item in event.neuron_ids),
                    event.voltages,
                    strict=True,
                )
            )
        external = {
            step: (
                np.asarray([index for index, _ in pairs], dtype=np.int64),
                np.asarray([voltage for _, voltage in pairs], dtype=np.float32),
            )
            for step, pairs in scheduled.items()
        }
        batches = tuple(
            simulate_shiu(
                graph,
                learning.shiu_parameters,
                steps=learning.neural_chunk_steps,
                external_voltage_events=external,
                state=state,
                silenced=silenced,
                presynaptic_transmitter_multipliers=(
                    learning.presynaptic_transmitter_multipliers
                ),
                plastic_edge_indices=binding.overlay.edge_indices,
                plastic_edge_multipliers=binding.overlay.multipliers,
            )
        )
        fired_ids = tuple(
            int(neuron_id) for batch in batches for neuron_id in batch.neuron_ids
        )
        plastic_kc_spikes += sum(neuron_id in plastic_kc_ids for neuron_id in fired_ids)
        mbon_spikes += sum(neuron_id in mbon_ids for neuron_id in fired_ids)
        for neuron_id in fired_ids:
            if neuron_id in dan_spike_counts:
                dan_spike_counts[neuron_id] += 1
            if neuron_id in mbon_spike_counts:
                mbon_spike_counts[neuron_id] += 1
        for name, ids in descending_ids.items():
            descending_counts[name] += sum(neuron_id in ids for neuron_id in fired_ids)
        for neuron_id in fired_ids:
            group = motor_owner.get(neuron_id)
            if group is not None:
                motor_spikes += 1
                active_groups.add(group)
        phases = cast(
            tuple[float, float, float, float, float, float],
            tuple(leg.phase for leg in body.legs),
        )
        activation = decoder.decode(
            fired_ids,
            window_s=body_parameters.dt_s,
            leg_phases=phases,
        )
        torque_queue.append(activation.torques)
        if len(torque_queue) <= perturbation.delay_steps:
            applied_torque = HexapodTorque.zero()
        else:
            applied_torque = torque_queue[-(perturbation.delay_steps + 1)]
        if perturbation.damaged_legs:
            values = list(applied_torque.values)
            for leg_index in perturbation.damaged_legs:
                values[leg_index] = (0.0, 0.0, 0.0)
            applied_torque = HexapodTorque(values=tuple(values))
        next_body = backend.step(applied_torque)
        body_trace.append(next_body.model_dump(mode="json"))
        distance_to_food.append(_distance(next_body, config.arena.food_position_m))
        distance_to_threat.append(_distance(next_body, config.arena.threat_position_m))
        food_contact = (
            _distance(next_body, config.arena.food_position_m)
            <= config.arena.contact_radius_m
        )
        threat_contact = (
            _distance(next_body, config.arena.threat_position_m)
            <= config.arena.contact_radius_m
        )
        appetitive_contacts += int(food_contact)
        aversive_contacts += int(threat_contact)
        if food_contact and first_food_contact_step is None:
            first_food_contact_step = step_index
        if threat_contact and first_threat_contact_step is None:
            first_threat_contact_step = step_index
        if (
            first_threat_contact_step is not None
            and time_to_clear_threat_steps is None
            and step_index > first_threat_contact_step
            and not threat_contact
        ):
            time_to_clear_threat_steps = step_index - first_threat_contact_step
        recruitment = (
            reinforcement.recruit(
                AnonymousContact(
                    appetitive_intensity=1.0 if food_contact else 0.0,
                    aversive_intensity=1.0 if threat_contact else 0.0,
                )
            )
            if dan_enabled
            else reinforcement.recruit(AnonymousContact())
        )
        dan_events += int(bool(recruitment.dan_ids))
        observed_dan_ids = tuple(
            sorted(set(fired_ids) & dan_ids)
        )
        routed_dan_ids = (
            recruitment.dan_ids
            if config.reinforcement_source == "contact_recruited"
            else tuple(sorted(set(observed_dan_ids) & set(recruitment.dan_ids)))
        )
        routed_dan_spike_events += int(bool(routed_dan_ids))
        fired = np.asarray(fired_ids, dtype=np.uint64)
        if plasticity_enabled:
            learner.step(
                active_kc_ids=fired[np.isin(fired, binding.pre_ids)],
                active_mbon_ids=fired[np.isin(fired, binding.post_ids)],
                routed_dan_ids=np.asarray(routed_dan_ids, dtype=np.uint64),
                dt_ms=learning.neural_chunk_steps * learning.shiu_parameters.dt_ms,
            )
        if slow_state is not None and plasticity_enabled:
            routed_dans = set(routed_dan_ids)
            routed_posts = np.asarray(
                [
                    post_id
                    for dan_id, post_id in learning.dan_to_mbon_pairs
                    if dan_id in routed_dans
                ],
                dtype=np.uint64,
            )
            active_kc_edges = np.isin(binding.pre_ids, fired)
            routed_edges = np.isin(binding.post_ids, routed_posts)
            slow_state.step(
                paired=active_kc_edges & routed_edges,
                dan_unpaired=(~active_kc_edges) & routed_edges,
                dt_seconds=neural_window_s,
                parameters=config.slow_memory_parameters,
            )
            slow_multipliers = np.asarray(
                learner.overlay.multipliers * slow_state.weight_multipliers,
                dtype=np.float32,
            )
            binding.overlay.set_multipliers(
                slow_multipliers,
                minimum=0.0,
                maximum=2.0,
            )
    final_body = backend.observe()
    dopamine_effect_max = (
        float(np.max(slow_state.dopamine_effect)) if slow_state is not None else 0.0
    )
    nitric_oxide_effect_max = (
        float(np.max(slow_state.nitric_oxide_effect)) if slow_state is not None else 0.0
    )
    multiplier_buckets: dict[int, list[float]] = {}
    for post_id, multiplier in zip(
        binding.post_ids.tolist(), binding.overlay.multipliers.tolist(), strict=True
    ):
        multiplier_buckets.setdefault(int(post_id), []).append(float(multiplier))
    mbon_mean_multipliers = {
        post_id: float(np.mean(values))
        for post_id, values in sorted(multiplier_buckets.items())
    }
    return _Trace(
        backend=backend.identity,
        appetitive_contacts=appetitive_contacts,
        aversive_contacts=aversive_contacts,
        reinforcement_source=config.reinforcement_source,
        dan_events=dan_events,
        routed_dan_spike_events=routed_dan_spike_events,
        dan_spike_counts=dan_spike_counts,
        plastic_kc_spikes=plastic_kc_spikes,
        mbon_spikes=mbon_spikes,
        mbon_spike_counts=mbon_spike_counts,
        mbon_mean_multipliers=mbon_mean_multipliers,
        descending_spikes=DescendingSpikeSummary(
            d_na02_left=descending_counts.get("d_na02_left", 0),
            d_na02_right=descending_counts.get("d_na02_right", 0),
            d_ng13_left=descending_counts.get("d_ng13_left", 0),
            d_ng13_right=descending_counts.get("d_ng13_right", 0),
            mdn_left=descending_counts.get("mdn_left", 0),
            mdn_right=descending_counts.get("mdn_right", 0),
        ),
        motor_spikes=motor_spikes,
        active_motor_groups=tuple(sorted(active_groups)),
        proprioceptive_events=proprioceptive_events,
        tactile_contact_events=tactile_contact_events,
        visual_source_events=visual_source_events,
        slow_memory_enabled=slow_state is not None,
        slow_memory_edges=(
            int(np.count_nonzero(slow_state.nitric_oxide_competent))
            if slow_state is not None
            else 0
        ),
        slow_memory_dopamine_effect_max=dopamine_effect_max,
        slow_memory_nitric_oxide_effect_max=nitric_oxide_effect_max,
        final_multipliers=tuple(float(value) for value in binding.overlay.multipliers),
        learning_memory=AutonomousLearningMemory.capture(
            learner, slow_state, binding.overlay.multipliers, memory_context,
        ),
        initial_food_distance=initial_food_distance,
        initial_threat_distance=initial_threat_distance,
        trace_distance_to_food=tuple(distance_to_food),
        trace_distance_to_threat=tuple(distance_to_threat),
        first_food_contact_step=first_food_contact_step,
        first_threat_contact_step=first_threat_contact_step,
        time_to_clear_threat_steps=time_to_clear_threat_steps,
        final_body=final_body,
        body_digest=hashlib.sha256(repr(body_trace).encode("utf-8")).hexdigest(),
    )


def run_autonomous_hexapod_episode(
    graph: EventConnectome,
    binding: PlasticEdgeBinding,
    config: AutonomousHexapodConfig,
    *,
    motor: HexapodMotorMap,
    proprio: ProprioceptiveMap,
    reinforcement: ReinforcementInterface,
    body_parameters: HexapodParameters,
    backend_factory: BackendFactory = ReferenceHexapodBackend,
    proprioceptive_calibration: ProprioceptiveCalibration | None = None,
    perturbation: BodyPerturbation | None = None,
    replay: bool = True,
    mutate_binding: bool = False,
    dan_enabled: bool = True,
    plasticity_enabled: bool = True,
    learning_memory: AutonomousLearningMemory | None = None,
) -> AutonomousHexapodResult:
    """Run and replay a schedule-free physical-contact learning episode."""

    before = _graph_digest(graph)
    context = hashlib.sha256(before.encode())
    context.update(config.learning.model_dump_json().encode())
    context.update(repr((
        config.nitric_oxide_dan_ids, config.slow_memory_parameters,
        config.reinforcement_source, reinforcement, graph.transmitters,
    )).encode())
    for array in (binding.pre_ids, binding.post_ids, binding.overlay.edge_indices):
        context.update(str(array.shape).encode())
        context.update(array.tobytes())
    memory_context = context.hexdigest()
    calibration = proprioceptive_calibration or ProprioceptiveCalibration()
    active_perturbation = perturbation or BodyPerturbation()
    active_parameters = apply_perturbation(body_parameters, active_perturbation)
    approach_mbon_ids = _approach_mbon_ids(config.learning, reinforcement)
    replay_binding = _clone_binding(binding) if replay else None
    first_binding = binding if mutate_binding else _clone_binding(binding)
    first = _run(
        graph,
        first_binding,
        config,
        motor=motor,
        proprio=proprio,
        reinforcement=reinforcement,
        body_parameters=active_parameters,
        backend_factory=backend_factory,
        proprioceptive_calibration=calibration,
        perturbation=active_perturbation,
        dan_enabled=dan_enabled,
        plasticity_enabled=plasticity_enabled,
        learning_memory=learning_memory,
        memory_context=memory_context,
    )
    replay_trace = None
    if replay_binding is not None:
        replay_trace = _run(
            graph,
            replay_binding,
            config,
            motor=motor,
            proprio=proprio,
            reinforcement=reinforcement,
            body_parameters=active_parameters,
            backend_factory=backend_factory,
            proprioceptive_calibration=calibration,
            perturbation=active_perturbation,
            dan_enabled=dan_enabled,
            plasticity_enabled=plasticity_enabled,
            learning_memory=learning_memory,
            memory_context=memory_context,
        )
    return AutonomousHexapodResult(
        odor_channel_model=(
            "measured_bilateral_receptor_channels"
            if config.odor_a_left_input_ids
            else "receptor_bank_olfactory_assumption"
            if config.odor_a_input_ids
            else "bipartite_registered_olfactory_assumption"
        ),
        proprioceptive_encoding=config.proprioceptive_encoding,
        proprioceptive_spike_rate_hz=config.proprioceptive_spike_rate_hz,
        phase_envelope=config.phase_envelope,
        neural_authority_nm=config.neural_authority_nm,
        plasticity_mode=config.learning.learning_parameters.plasticity_mode,
        valence_compartment_model=(
            "dan_route_majority_assumption"
            if config.learning.learning_parameters.plasticity_mode
            == "valence_compartmental"
            else "none"
        ),
        approach_mbon_count=int(approach_mbon_ids.size),
        tactile_contact_model=(
            "uniform_registered_vnc_tactile_assumption"
            if config.tactile_contact_input_ids
            else "none"
        ),
        backend=first.backend,
        replay_performed=replay,
        replay_exact=replay and first == replay_trace,
        graph_unchanged=_graph_digest(graph) == before,
        body_steps=config.body_steps,
        neural_steps=config.body_steps * config.learning.neural_chunk_steps,
        appetitive_contacts=first.appetitive_contacts,
        aversive_contacts=first.aversive_contacts,
        reinforcement_source=first.reinforcement_source,
        dan_events=first.dan_events,
        routed_dan_spike_events=first.routed_dan_spike_events,
        dan_spike_counts=first.dan_spike_counts,
        plastic_kc_spikes=first.plastic_kc_spikes,
        mbon_spikes=first.mbon_spikes,
        mbon_spike_counts=first.mbon_spike_counts,
        mbon_mean_multipliers=first.mbon_mean_multipliers,
        descending_spikes=first.descending_spikes,
        motor_spikes=first.motor_spikes,
        active_motor_groups=len(first.active_motor_groups),
        all_motor_groups_active=(
            len(first.active_motor_groups) == len(CANONICAL_MOTOR_GROUPS)
        ),
        proprioceptive_events=first.proprioceptive_events,
        tactile_contact_events=first.tactile_contact_events,
        visual_source_events=first.visual_source_events,
        slow_memory_enabled=first.slow_memory_enabled,
        slow_memory_edges=first.slow_memory_edges,
        slow_memory_dopamine_effect_max=first.slow_memory_dopamine_effect_max,
        slow_memory_nitric_oxide_effect_max=(
            first.slow_memory_nitric_oxide_effect_max
        ),
        final_multipliers=first.final_multipliers,
        learning_memory=first.learning_memory,
        initial_food_distance=first.initial_food_distance,
        initial_threat_distance=first.initial_threat_distance,
        trace_distance_to_food=first.trace_distance_to_food,
        trace_distance_to_threat=first.trace_distance_to_threat,
        first_food_contact_step=first.first_food_contact_step,
        first_threat_contact_step=first.first_threat_contact_step,
        time_to_clear_threat_steps=first.time_to_clear_threat_steps,
        final_body=first.final_body,
        trace_digest=first.digest,
    )
