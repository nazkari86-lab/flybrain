import os
from pathlib import Path

import pytest

from flybrain.autonomous_learning_benchmark import (
    AssociativeCalibrationConfig,
    run_associative_calibration,
)
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.conditioning_world import ConditioningSchedule
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters

REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")


def test_retained_malecns_direct_cue_dan_calibration() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(raw_snapshot)
    registry = load_biological_registry(REGISTRY)
    populations = resolve_biological_registry(registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(registry, populations, snapshot)
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(populations)
    result = run_associative_calibration(
        graph,
        binding,
        schedule=ConditioningSchedule.create(
            seed=7, steps=1, appetitive_pair_steps=(0,)
        ),
        reinforcement=reinforcement,
        config=AssociativeCalibrationConfig.from_manifest(
            binding,
            manifests["dan_to_mbon"],
            reinforcement,
            valence="appetitive",
            neural_chunk_steps=8,
            shiu_parameters=ShiuParameters(
                dt_ms=1.0,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.classification in {"plasticity_calibration", "null"}
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert len(result.final_multipliers) == 33_496


def test_retained_malecns_olfactory_path_dan_calibration() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(raw_snapshot)
    registry = load_biological_registry(REGISTRY)
    populations = resolve_biological_registry(registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(registry, populations, snapshot)
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(populations)
    result = run_associative_calibration(
        graph,
        binding,
        schedule=ConditioningSchedule.create(
            seed=7, steps=1, appetitive_pair_steps=(0,)
        ),
        reinforcement=reinforcement,
        config=AssociativeCalibrationConfig.from_manifest(
            binding,
            manifests["dan_to_mbon"],
            reinforcement,
            valence="appetitive",
            sensory_input_ids=populations.population("olfactory_sensory").neuron_ids,
            neural_chunk_steps=30,
            shiu_parameters=ShiuParameters(
                dt_ms=1.0,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
    )

    assert result.input_mode == "sensory_path"
    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.classification in {"plasticity_calibration", "null"}
    assert result.cue_spikes > 0
    assert result.mbon_spikes > 0
    assert result.replay_exact is True
    assert result.graph_unchanged is True
