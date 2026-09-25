import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flybrain.associative_motor_pathway import resolve_mbon_motor_pathway
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS, HexapodMotorMap, MotorGroup


def motor_map() -> HexapodMotorMap:
    return HexapodMotorMap(
        groups=tuple(
            MotorGroup(
                name=name,
                leg=leg,
                joint=joint,
                direction=direction,
                neuron_ids=(100 + index,),
            )
            for index, (name, leg, joint, direction) in enumerate(
                CANONICAL_MOTOR_GROUPS
            )
        )
    )


def test_pathway_audit_records_only_exact_direct_and_two_hop_edges(
    tmp_path: Path,
) -> None:
    pq.write_table(
        pa.table(
            {
                "pre_id": np.array([10, 10, 11, 50, 51], dtype=np.uint64),
                "post_id": np.array([50, 101, 51, 100, 102], dtype=np.uint64),
                "synapse_count": np.array([7, 5, 3, 11, 13], dtype=np.int64),
                "sign": np.array([1, 1, -1, 1, -1], dtype=np.int8),
            }
        ),
        tmp_path / "edges.parquet",
    )

    result = resolve_mbon_motor_pathway(tmp_path, mbon_ids=(10, 11), motor=motor_map())

    assert result.evidence_kind == "dataset_measurement"
    assert result.direct_edge_count == 1
    assert result.direct_contact_count == 5
    assert result.two_hop_edge_count == 2
    assert result.two_hop_route_count == 2
    assert result.two_hop_intermediate_count == 2
    assert result.two_hop_contact_count == 24
    assert tuple(
        (item.mbon_id, item.intermediate_id, item.motor_id) for item in result.routes
    ) == ((10, 50, 100), (11, 51, 102))
    assert tuple(item.signs for item in result.routes) == ((1, 1), (-1, -1))


def test_retained_malecns_audit_keeps_mbon_motor_anatomy_explicit() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(raw_snapshot)
    learning = resolve_biological_registry(
        load_biological_registry(
            Path("data/registry/autonomous-learning-registry-v1.json")
        ),
        snapshot,
    )
    motor_registry = resolve_biological_registry(
        load_biological_registry(Path("data/registry/hexapod-motor-registry-v2.json")),
        snapshot,
    )

    result = resolve_mbon_motor_pathway(
        snapshot,
        mbon_ids=learning.population("mbons").neuron_ids,
        motor=HexapodMotorMap.from_registry(motor_registry),
    )

    assert result.evidence_kind == "dataset_measurement"
    assert result.direct_edge_count == 0
    assert result.direct_contact_count == 0
    assert result.two_hop_edge_count == 60
    assert result.two_hop_route_count == 108
    assert result.two_hop_intermediate_count == 16
    assert result.two_hop_contact_count == 1502
    assert result.route_digest == (
        "44f4383a89f1be558016e024e1fbce8c25cc49beb0db82ce750e52cc917e8e55"
    )

    old_motor_registry = resolve_biological_registry(
        load_biological_registry(Path("data/registry/hexapod-motor-registry-v1.json")),
        snapshot,
    )
    old_motor_ids = {
        neuron_id
        for population in old_motor_registry.populations
        if population.role == "motor"
        for neuron_id in population.neuron_ids
    }
    current_motor_ids = {
        neuron_id for group in HexapodMotorMap.from_registry(motor_registry).groups
        for neuron_id in group.neuron_ids
    }
    assert old_motor_ids <= current_motor_ids
    old_routes = tuple(route for route in result.routes if route.motor_id in old_motor_ids)
    assert len(old_routes) == 46
    assert len({(route.intermediate_id, route.motor_id) for route in old_routes}) == 31
