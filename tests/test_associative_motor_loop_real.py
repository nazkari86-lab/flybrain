import os
from pathlib import Path

import pytest

from flybrain.associative_motor_loop import run_associative_motor_calibration
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.conditioning_world import ConditioningEvent, ConditioningSchedule
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


def test_retained_malecns_associative_motor_body_loop_is_replayable() -> None:
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
    config = AssociativeCalibrationConfig.from_manifest(
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
    motor = HexapodMotorMap.from_registry(motor_populations)
    proprio = ProprioceptiveMap.from_registry(motor_populations)
    paired_schedule = ConditioningSchedule.create(
        seed=7,
        steps=1,
        appetitive_pair_steps=(0,),
    )
    no_contact_schedule = ConditioningSchedule(
        seed=7,
        events=(
            ConditioningEvent(
                step=0,
                odor_intensity=1.0,
                appetitive_contact_intensity=0.0,
                aversive_contact_intensity=0.0,
            ),
        ),
    )
    common = dict(
        graph=graph,
        binding=binding,
        learning=config,
        motor=motor,
        proprio=proprio,
        reinforcement=reinforcement,
        body_parameters=HexapodParameters(dt_s=0.01),
    )
    result = run_associative_motor_calibration(
        schedule=paired_schedule,
        **common,
    )
    motor_lesion = run_associative_motor_calibration(
        schedule=paired_schedule,
        motor_silenced=True,
        **common,
    )
    no_contact = run_associative_motor_calibration(
        schedule=no_contact_schedule,
        **common,
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.proprioceptive_events == 6
    assert len(result.final_multipliers) == 33_496
    assert motor_lesion.motor_spikes == 0
    assert motor_lesion.proprioceptive_events == result.proprioceptive_events
    assert motor_lesion.graph_unchanged is True
    assert no_contact.final_multipliers == (1.0,) * 33_496
