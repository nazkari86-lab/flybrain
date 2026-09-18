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
    cue_ids = tuple(sorted(set(int(value) for value in binding.pre_ids)))
    mbon_ids = tuple(sorted(set(int(value) for value in binding.post_ids)))
    allowed_appetitive = set(reinforcement.appetitive_dan_ids)
    dan_routes = tuple(
        (pre_id, post_id)
        for pre_id, post_id in manifests["dan_to_mbon"].edge_pairs
        if pre_id in allowed_appetitive and post_id in set(mbon_ids)
    )
    result = run_associative_calibration(
        graph,
        binding,
        schedule=ConditioningSchedule.create(
            seed=7, steps=1, appetitive_pair_steps=(0,)
        ),
        reinforcement=reinforcement,
        config=AssociativeCalibrationConfig(
            cue_ids=cue_ids,
            mbon_ids=mbon_ids,
            dan_to_mbon_pairs=dan_routes,
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
