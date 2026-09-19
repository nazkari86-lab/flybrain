import os
from pathlib import Path

import pytest

from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    HexapodArenaConfig,
    run_autonomous_hexapod_episode,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters

LEARNING_REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
MOTOR_REGISTRY = Path("data/registry/hexapod-motor-registry-v1.json")


def test_real_malecns_runs_contact_driven_hexapod_without_schedule() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(raw_snapshot)
    learning_registry = load_biological_registry(LEARNING_REGISTRY)
    learning_populations = resolve_biological_registry(learning_registry, snapshot)
    motor_populations = resolve_biological_registry(
        load_biological_registry(MOTOR_REGISTRY), snapshot
    )
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
    learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=learning_populations.population("olfactory_sensory").neuron_ids,
        neural_chunk_steps=100,
        shiu_parameters=ShiuParameters(
            dt_ms=0.1,
            refractory_ms=2.0,
            synaptic_delay_ms=1.0,
        ),
    )
    aversive_learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="aversive",
        sensory_input_ids=learning_populations.population("olfactory_sensory").neuron_ids,
        neural_chunk_steps=100,
        shiu_parameters=learning.shiu_parameters,
    )
    bivalent_learning = learning.model_copy(
        update={
            "dan_to_mbon_pairs": tuple(
                sorted(
                    set(learning.dan_to_mbon_pairs)
                    | set(aversive_learning.dan_to_mbon_pairs)
                )
            )
        }
    )

    result = run_autonomous_hexapod_episode(
        graph,
        binding,
        AutonomousHexapodConfig(
            body_steps=2,
            learning=bivalent_learning,
            arena=HexapodArenaConfig(
                food_position_m=(0.0, 0.0),
                threat_position_m=(10.0, 10.0),
                contact_radius_m=0.05,
                odor_length_scale_m=1.0,
            ),
            seed=7,
        ),
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=ProprioceptiveMap.from_registry(motor_populations),
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert graph.neuron_count == 166_606
    assert graph.edge_count == 6_240_402
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.appetitive_contacts == 2
    assert result.dan_events == 2
    assert result.motor_spikes > 0
    assert result.proprioceptive_events == 12
    assert len(result.final_multipliers) == 33_496
    assert any(value < 1.0 for value in result.final_multipliers)

    threat_result = run_autonomous_hexapod_episode(
        graph,
        binding,
        AutonomousHexapodConfig(
            body_steps=2,
            learning=bivalent_learning,
            arena=HexapodArenaConfig(
                food_position_m=(10.0, 10.0),
                threat_position_m=(0.0, 0.0),
                contact_radius_m=0.05,
                odor_length_scale_m=1.0,
            ),
            seed=7,
        ),
        motor=HexapodMotorMap.from_registry(motor_populations),
        proprio=ProprioceptiveMap.from_registry(motor_populations),
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert threat_result.replay_exact is True
    assert threat_result.appetitive_contacts == 0
    assert threat_result.aversive_contacts == 2
    assert threat_result.dan_events == 2
    assert threat_result.motor_spikes > 0
    assert any(value < 1.0 for value in threat_result.final_multipliers)
