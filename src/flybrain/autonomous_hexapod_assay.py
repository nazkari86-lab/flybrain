"""Retained-MaleCNS launcher for the schedule-free autonomous hexapod loop."""

from __future__ import annotations

import math
import resource
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.autonomous_behavior_benchmark import (
    BehaviorBenchmarkConfig,
    BehaviorBenchmarkResult,
    run_behavior_benchmark,
)
from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    AutonomousHexapodResult,
    HexapodArenaConfig,
    run_autonomous_hexapod_episode,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_perturbations import BehaviorVariant, BodyPerturbation
from flybrain.biological_registry import (
    ResolvedRegistry,
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.descending_interface import DescendingMap
from flybrain.flygym_backend import FlyGymBackend
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_backend import ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.learning_memory import AutonomousLearningMemory
from flybrain.mb_association import _software_revision
from flybrain.mushroom_body_learning import MushroomBodyLearningParameters
from flybrain.olfactory_interface import OlfactoryReceptorMap, task_odor_assignment
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.proprioceptive_subtypes import load_proprioceptive_subtypes
from flybrain.provenance import snapshot_content_sha256
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.retinal_interface import VisualInterfaceMap
from flybrain.shiu import ShiuParameters
from flybrain.slow_memory import resolve_nitric_oxide_dans

DEFAULT_LEARNING_REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
DEFAULT_MOTOR_REGISTRY = Path("data/registry/hexapod-motor-registry-v2.json")
DEFAULT_INTERFACE_REGISTRY = Path("data/registry/biological-interface-registry-v1.json")


def _visual_interface(populations: ResolvedRegistry) -> VisualInterfaceMap:
    return VisualInterfaceMap(
        left_r1_r6_ids=populations.population("visual_r1_r6_left").neuron_ids,
        right_r1_r6_ids=populations.population("visual_r1_r6_right").neuron_ids,
        left_hs_ids=populations.population("hs_left").neuron_ids,
        right_hs_ids=populations.population("hs_right").neuron_ids,
        left_lc16_ids=populations.population("lc16_left").neuron_ids,
        right_lc16_ids=populations.population("lc16_right").neuron_ids,
    )


def _descending_map(populations: ResolvedRegistry) -> DescendingMap:
    return DescendingMap(
        d_na02_left=populations.population("d_na02_left").neuron_ids,
        d_na02_right=populations.population("d_na02_right").neuron_ids,
        d_ng13_left=populations.population("d_ng13_left").neuron_ids,
        d_ng13_right=populations.population("d_ng13_right").neuron_ids,
        mdn_left=populations.population("mdn_left").neuron_ids,
        mdn_right=populations.population("mdn_right").neuron_ids,
    )


class RetainedAutonomousHexapodAssay(BaseModel, frozen=True):
    """Immutable provenance and result payload for one retained-snapshot run."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["retained-autonomous-hexapod-assay-v1"] = (
        "retained-autonomous-hexapod-assay-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    learning_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    motor_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interface_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    appetitive_dan_routes: int = Field(gt=0)
    aversive_dan_routes: int = Field(gt=0)
    body_steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    software_revision: str = Field(min_length=1)
    runtime_seconds: float = Field(gt=0.0)
    peak_rss_bytes: int = Field(gt=0)
    episode: AutonomousHexapodResult


class RetainedAutonomousBehaviorBenchmark(BaseModel, frozen=True):
    """Provenance envelope for the multi-condition retained behavior benchmark."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["retained-autonomous-behavior-benchmark-v2"] = (
        "retained-autonomous-behavior-benchmark-v2"
    )
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    learning_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    motor_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interface_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    training_episodes: int = Field(gt=0)
    holdout_episodes: int = Field(gt=0)
    seeds: tuple[int, ...]
    software_revision: str = Field(min_length=1)
    runtime_seconds: float = Field(gt=0.0)
    peak_rss_bytes: int = Field(gt=0)
    benchmark: BehaviorBenchmarkResult


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


_FLYGYM_NEURAL_AUTHORITY_NM = (1e-5, 1e-6, 1e-6)
_FLYGYM_ARENA = HexapodArenaConfig(
    food_position_m=(0.0015, 0.0),
    threat_position_m=(-0.0015, 0.0),
    contact_radius_m=0.0002,
    odor_length_scale_m=0.002,
    antenna_lateral_offset_m=0.00015,
    visual_disc_radius_m=0.00015,
)


def run_retained_autonomous_hexapod_assay(
    snapshot: Path,
    *,
    learning_registry_path: Path = DEFAULT_LEARNING_REGISTRY,
    motor_registry_path: Path = DEFAULT_MOTOR_REGISTRY,
    interface_registry_path: Path = DEFAULT_INTERFACE_REGISTRY,
    backend: Literal["reference", "flygym"] = "reference",
    body_steps: int = 2,
    seed: int = 7,
    proprioceptive_spike_rate_hz: float = 150.0,
    proprioceptive_encoding: Literal[
        "population_voltage", "source_equivalent_spikes",
        "budget_matched_uniform_spikes", "subtype_weighted_spikes",
    ] = "source_equivalent_spikes",
    learning_memory: AutonomousLearningMemory | None = None,
    capture_motor_trace: bool = False,
    motor_lesion_groups: tuple[str, ...] = (),
) -> RetainedAutonomousHexapodAssay:
    """Resolve exact registries and run one autonomous contact-learning episode."""

    if type(body_steps) is not int or body_steps <= 0:
        raise ValueError("body_steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(proprioceptive_spike_rate_hz) or proprioceptive_spike_rate_hz <= 0.0:
        raise ValueError("proprioceptive spike rate must be finite and positive")
    started = time.perf_counter()
    learning_registry = load_biological_registry(learning_registry_path)
    motor_registry = load_biological_registry(motor_registry_path)
    interface_registry = load_biological_registry(interface_registry_path)
    learning_populations = resolve_biological_registry(learning_registry, snapshot)
    motor_populations = resolve_biological_registry(motor_registry, snapshot)
    interface_populations = resolve_biological_registry(interface_registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(
            learning_registry,
            learning_populations,
            snapshot,
        )
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(learning_populations)
    sensory_input_ids = learning_populations.population(
        "olfactory_sensory"
    ).neuron_ids
    shiu_parameters = ShiuParameters(
        dt_ms=0.1,
        refractory_ms=2.0,
        synaptic_delay_ms=1.0,
    )
    valence_learning_parameters = MushroomBodyLearningParameters(
        maximum_multiplier=2.0,
        plasticity_mode="valence_compartmental",
    )
    appetitive_learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
        learning_parameters=valence_learning_parameters,
    )
    aversive_learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="aversive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
        learning_parameters=valence_learning_parameters,
    )
    learning = appetitive_learning.model_copy(
        update={
            "dan_to_mbon_pairs": tuple(
                sorted(
                    set(appetitive_learning.dan_to_mbon_pairs)
                    | set(aversive_learning.dan_to_mbon_pairs)
                )
            )
        }
    )
    nitric_oxide_dans = resolve_nitric_oxide_dans(snapshot)
    if learning_memory is not None:
        binding.overlay.set_multipliers(
            np.asarray(learning_memory.effective_multipliers, dtype=np.float32),
            minimum=0.0, maximum=2.0,
        )
    backend_factory = FlyGymBackend if backend == "flygym" else ReferenceHexapodBackend
    proprio = ProprioceptiveMap.from_registry(motor_populations)
    subtypes = (
        load_proprioceptive_subtypes(snapshot / "source-annotations.parquet", proprio)
        if proprioceptive_encoding in {
            "budget_matched_uniform_spikes", "subtype_weighted_spikes"
        }
        else None
    )
    episode = run_autonomous_hexapod_episode(
        graph,
        binding,
        AutonomousHexapodConfig(
            body_steps=body_steps,
            learning=learning,
            arena=(
                _FLYGYM_ARENA if backend == "flygym" else HexapodArenaConfig(
                    food_position_m=(0.0, 0.0),
                    threat_position_m=(10.0, 10.0),
                    contact_radius_m=0.05,
                    odor_length_scale_m=1.0,
                )
            ),
            seed=seed,
            nitric_oxide_dan_ids=nitric_oxide_dans.all_ids,
            tactile_contact_input_ids=interface_populations.population(
                "tactile_vnc"
            ).neuron_ids,
            proprioceptive_encoding=proprioceptive_encoding,
            proprioceptive_spike_rate_hz=proprioceptive_spike_rate_hz,
            neural_authority_nm=(
                _FLYGYM_NEURAL_AUTHORITY_NM if backend == "flygym"
                else (0.002, 0.0002, 0.0002)
            ),
            reinforcement_source="contact_gated_neural_dan",
            visual_interface=_visual_interface(interface_populations),
            descending_map=_descending_map(interface_populations),
        ),
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=proprio,
        proprioceptive_subtypes=subtypes,
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
        backend_factory=backend_factory,
        learning_memory=learning_memory,
        capture_motor_trace=capture_motor_trace,
        motor_lesion_groups=motor_lesion_groups,
    )
    return RetainedAutonomousHexapodAssay(
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        learning_registry_sha256=learning_populations.registry_sha256,
        motor_registry_sha256=motor_populations.registry_sha256,
        interface_registry_sha256=interface_populations.registry_sha256,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        appetitive_dan_routes=len(appetitive_learning.dan_to_mbon_pairs),
        aversive_dan_routes=len(aversive_learning.dan_to_mbon_pairs),
        body_steps=body_steps,
        seed=seed,
        software_revision=_software_revision(),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        episode=episode,
    )


def run_retained_autonomous_behavior_benchmark(
    snapshot: Path,
    *,
    learning_registry_path: Path = DEFAULT_LEARNING_REGISTRY,
    motor_registry_path: Path = DEFAULT_MOTOR_REGISTRY,
    interface_registry_path: Path = DEFAULT_INTERFACE_REGISTRY,
    backend: Literal["reference", "flygym"] = "reference",
    training_episodes: int = 1,
    holdout_episodes: int = 1,
    body_steps: int = 2,
    seed: int = 7,
    seeds: tuple[int, ...] | None = None,
    proprioceptive_spike_rate_hz: float = 150.0,
    proprioceptive_encoding: Literal[
        "population_voltage", "source_equivalent_spikes",
        "budget_matched_uniform_spikes", "subtype_weighted_spikes",
    ] = "source_equivalent_spikes",
) -> RetainedAutonomousBehaviorBenchmark:
    """Run the bounded multi-condition benchmark on a retained snapshot."""

    if training_episodes <= 0 or holdout_episodes <= 0 or body_steps <= 0:
        raise ValueError("benchmark episode and body counts must be positive")
    active_seeds = tuple(seeds or (seed,))
    if not active_seeds or any(value < 0 for value in active_seeds):
        raise ValueError("seeds must be non-empty and non-negative")
    if not math.isfinite(proprioceptive_spike_rate_hz) or proprioceptive_spike_rate_hz <= 0.0:
        raise ValueError("proprioceptive spike rate must be finite and positive")
    started = time.perf_counter()
    learning_registry = load_biological_registry(learning_registry_path)
    motor_registry = load_biological_registry(motor_registry_path)
    interface_registry = load_biological_registry(interface_registry_path)
    learning_populations = resolve_biological_registry(learning_registry, snapshot)
    motor_populations = resolve_biological_registry(motor_registry, snapshot)
    interface_populations = resolve_biological_registry(interface_registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(
            learning_registry, learning_populations, snapshot
        )
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    proprio = ProprioceptiveMap.from_registry(motor_populations)
    subtypes = (
        load_proprioceptive_subtypes(snapshot / "source-annotations.parquet", proprio)
        if proprioceptive_encoding in {
            "budget_matched_uniform_spikes", "subtype_weighted_spikes"
        }
        else None
    )
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(learning_populations)
    sensory_input_ids = learning_populations.population("olfactory_sensory").neuron_ids
    receptor_map = OlfactoryReceptorMap.from_snapshot(
        snapshot,
        graph=graph,
        olfactory_neuron_ids=sensory_input_ids,
    )
    odor_assignment = task_odor_assignment()
    odor_a_left_input_ids = receptor_map.side_channel_ids(
        odor_assignment.food_cell_types, "L"
    )
    odor_a_right_input_ids = receptor_map.side_channel_ids(
        odor_assignment.food_cell_types, "R"
    )
    odor_b_left_input_ids = receptor_map.side_channel_ids(
        odor_assignment.threat_cell_types, "L"
    )
    odor_b_right_input_ids = receptor_map.side_channel_ids(
        odor_assignment.threat_cell_types, "R"
    )
    shiu_parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    valence_learning_parameters = MushroomBodyLearningParameters(
        maximum_multiplier=2.0,
        plasticity_mode="valence_compartmental",
    )
    appetitive = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
        learning_parameters=valence_learning_parameters,
    )
    aversive = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="aversive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
        learning_parameters=valence_learning_parameters,
    )
    learning = appetitive.model_copy(
        update={
            "dan_to_mbon_pairs": tuple(
                sorted(set(appetitive.dan_to_mbon_pairs) | set(aversive.dan_to_mbon_pairs))
            )
        }
    )
    no_dans = resolve_nitric_oxide_dans(snapshot)
    food_training = BehaviorVariant(
        name="retained-food-training",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    threat_training = BehaviorVariant(
        name="retained-threat-training",
        food_position_m=(10.0, 10.0),
        threat_position_m=(0.0, 0.0),
    )
    food_holdout_a = BehaviorVariant(
        name="retained-food-holdout-a",
        evaluation_target="food",
        food_position_m=(0.25, 0.15),
        threat_position_m=(9.5, 9.5),
    )
    food_holdout_b = BehaviorVariant(
        name="retained-food-holdout-b",
        evaluation_target="food",
        food_position_m=(-0.25, 0.15),
        threat_position_m=(-9.5, 9.5),
        initial_position_m=(0.02, -0.01),
        perturbation=BodyPerturbation(
            friction_scale=0.9, mass_scale=1.05, delay_steps=1
        ),
    )
    threat_holdout_a = BehaviorVariant(
        name="retained-threat-holdout-a",
        evaluation_target="threat",
        food_position_m=(10.0, 10.0),
        threat_position_m=(0.10, 0.0),
    )
    threat_holdout_b = BehaviorVariant(
        name="retained-threat-holdout-b",
        evaluation_target="threat",
        food_position_m=(-10.0, 10.0),
        threat_position_m=(-0.10, 0.0),
        initial_position_m=(-0.01, 0.02),
        perturbation=BodyPerturbation(
            friction_scale=1.1, mass_scale=0.95
        ),
    )
    if backend == "flygym":
        food_training = food_training.model_copy(update={
            "food_position_m": (0.0005, 0.0),
            "threat_position_m": (-0.003, 0.0),
        })
        threat_training = threat_training.model_copy(update={
            "food_position_m": (0.003, 0.0),
            "threat_position_m": (0.0005, 0.0),
        })
        food_holdout_a = food_holdout_a.model_copy(update={
            "food_position_m": (0.0015, 0.0004),
            "threat_position_m": (-0.003, 0.003),
        })
        food_holdout_b = food_holdout_b.model_copy(update={
            "food_position_m": (-0.0015, 0.0004),
            "threat_position_m": (0.003, 0.003),
            "initial_position_m": (0.0002, -0.0001),
        })
        threat_holdout_a = threat_holdout_a.model_copy(update={
            "food_position_m": (0.003, 0.003),
            "threat_position_m": (0.001, 0.0),
        })
        threat_holdout_b = threat_holdout_b.model_copy(update={
            "food_position_m": (-0.003, 0.003),
            "threat_position_m": (-0.001, 0.0),
            "initial_position_m": (-0.0001, 0.0002),
        })
    config = BehaviorBenchmarkConfig(
        training_episodes=training_episodes,
        holdout_episodes=holdout_episodes,
        body_steps=body_steps,
        seeds=active_seeds,
        training_variants=(food_training, threat_training),
        holdout_variants=(
            food_holdout_a,
            food_holdout_b,
            threat_holdout_a,
            threat_holdout_b,
        ),
        nitric_oxide_dan_ids=no_dans.all_ids,
        tactile_contact_input_ids=interface_populations.population("tactile_vnc").neuron_ids,
        odor_a_left_input_ids=odor_a_left_input_ids,
        odor_a_right_input_ids=odor_a_right_input_ids,
        odor_b_left_input_ids=odor_b_left_input_ids,
        odor_b_right_input_ids=odor_b_right_input_ids,
        proprioceptive_encoding=proprioceptive_encoding,
        proprioceptive_spike_rate_hz=proprioceptive_spike_rate_hz,
        contact_radius_m=(
            _FLYGYM_ARENA.contact_radius_m if backend == "flygym" else 0.05
        ),
        odor_length_scale_m=(
            _FLYGYM_ARENA.odor_length_scale_m if backend == "flygym" else 1.0
        ),
        antenna_lateral_offset_m=(
            _FLYGYM_ARENA.antenna_lateral_offset_m if backend == "flygym" else 0.02
        ),
        visual_disc_radius_m=(
            _FLYGYM_ARENA.visual_disc_radius_m if backend == "flygym" else 0.05
        ),
        neural_authority_nm=(
            _FLYGYM_NEURAL_AUTHORITY_NM if backend == "flygym"
            else (0.002, 0.0002, 0.0002)
        ),
        reinforcement_source="contact_gated_neural_dan",
        visual_interface=_visual_interface(interface_populations),
        descending_map=_descending_map(interface_populations),
        odor_assignment=odor_assignment,
    )
    backend_factory = FlyGymBackend if backend == "flygym" else ReferenceHexapodBackend
    benchmark = run_behavior_benchmark(
        graph,
        binding,
        config,
        learning=learning,
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=proprio,
        proprioceptive_subtypes=subtypes,
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
        backend_factory=backend_factory,
    )
    return RetainedAutonomousBehaviorBenchmark(
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        learning_registry_sha256=learning_populations.registry_sha256,
        motor_registry_sha256=motor_populations.registry_sha256,
        interface_registry_sha256=interface_populations.registry_sha256,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        training_episodes=training_episodes,
        holdout_episodes=holdout_episodes,
        seeds=active_seeds,
        software_revision=_software_revision(),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        benchmark=benchmark,
    )
