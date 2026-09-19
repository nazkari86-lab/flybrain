"""Retained-MaleCNS launcher for the schedule-free autonomous hexapod loop."""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Literal

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
from flybrain.behavioral_perturbations import BehaviorVariant
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.flygym_backend import FlyGymBackend
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_backend import ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.mb_association import _software_revision
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.provenance import snapshot_content_sha256
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters
from flybrain.slow_memory import resolve_nitric_oxide_dans

DEFAULT_LEARNING_REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
DEFAULT_MOTOR_REGISTRY = Path("data/registry/hexapod-motor-registry-v1.json")
DEFAULT_INTERFACE_REGISTRY = Path("data/registry/biological-interface-registry-v1.json")


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

    protocol: Literal["retained-autonomous-behavior-benchmark-v1"] = (
        "retained-autonomous-behavior-benchmark-v1"
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


def run_retained_autonomous_hexapod_assay(
    snapshot: Path,
    *,
    learning_registry_path: Path = DEFAULT_LEARNING_REGISTRY,
    motor_registry_path: Path = DEFAULT_MOTOR_REGISTRY,
    interface_registry_path: Path = DEFAULT_INTERFACE_REGISTRY,
    backend: Literal["reference", "flygym"] = "reference",
    body_steps: int = 2,
    seed: int = 7,
) -> RetainedAutonomousHexapodAssay:
    """Resolve exact registries and run one autonomous contact-learning episode."""

    if type(body_steps) is not int or body_steps <= 0:
        raise ValueError("body_steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
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
    appetitive_learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
    )
    aversive_learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="aversive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
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
    backend_factory = FlyGymBackend if backend == "flygym" else ReferenceHexapodBackend
    episode = run_autonomous_hexapod_episode(
        graph,
        binding,
        AutonomousHexapodConfig(
            body_steps=body_steps,
            learning=learning,
            arena=HexapodArenaConfig(
                food_position_m=(0.0, 0.0),
                threat_position_m=(10.0, 10.0),
                contact_radius_m=0.05,
                odor_length_scale_m=1.0,
            ),
            seed=seed,
            nitric_oxide_dan_ids=nitric_oxide_dans.all_ids,
            tactile_contact_input_ids=interface_populations.population(
                "tactile_vnc"
            ).neuron_ids,
        ),
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=ProprioceptiveMap.from_registry(motor_populations),
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
        backend_factory=backend_factory,
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
) -> RetainedAutonomousBehaviorBenchmark:
    """Run the bounded multi-condition benchmark on a retained snapshot."""

    if training_episodes <= 0 or holdout_episodes <= 0 or body_steps <= 0:
        raise ValueError("benchmark episode and body counts must be positive")
    active_seeds = tuple(seeds or (seed,))
    if not active_seeds or any(value < 0 for value in active_seeds):
        raise ValueError("seeds must be non-empty and non-negative")
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
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(learning_populations)
    sensory_input_ids = learning_populations.population("olfactory_sensory").neuron_ids
    shiu_parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    appetitive = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
    )
    aversive = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="aversive",
        sensory_input_ids=sensory_input_ids,
        neural_chunk_steps=100,
        shiu_parameters=shiu_parameters,
    )
    learning = appetitive.model_copy(
        update={
            "dan_to_mbon_pairs": tuple(
                sorted(set(appetitive.dan_to_mbon_pairs) | set(aversive.dan_to_mbon_pairs))
            )
        }
    )
    no_dans = resolve_nitric_oxide_dans(snapshot)
    variant = BehaviorVariant(
        name="retained-training",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    holdout = variant.model_copy(
        update={
            "name": "retained-holdout",
            "food_position_m": (0.25, 0.15),
            "threat_position_m": (9.5, 9.5),
        }
    )
    threat_holdout = BehaviorVariant(
        name="retained-threat-holdout",
        food_position_m=(10.0, 10.0),
        threat_position_m=(0.0, 0.0),
    )
    config = BehaviorBenchmarkConfig(
        training_episodes=training_episodes,
        holdout_episodes=holdout_episodes,
        body_steps=body_steps,
        seeds=active_seeds,
        training_variants=(variant,),
        holdout_variants=(holdout, threat_holdout),
        nitric_oxide_dan_ids=no_dans.all_ids,
        tactile_contact_input_ids=interface_populations.population("tactile_vnc").neuron_ids,
    )
    backend_factory = FlyGymBackend if backend == "flygym" else ReferenceHexapodBackend
    benchmark = run_behavior_benchmark(
        graph,
        binding,
        config,
        learning=learning,
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=ProprioceptiveMap.from_registry(motor_populations),
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
