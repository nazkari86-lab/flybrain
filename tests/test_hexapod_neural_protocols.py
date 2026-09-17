import numpy as np
from scipy.sparse import csr_array

from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import LEG_NAMES
from flybrain.hexapod_motor import HexapodMotorMap, MotorGroup
from flybrain.hexapod_neural_protocols import run_dn_to_motor_protocols


def fixture(*, connected: bool) -> tuple[EventConnectome, DescendingMap, HexapodMotorMap]:
    descending = DescendingMap((1,), (2,), (3,), (4,), (5,), (6,))
    groups = []
    next_id = 100
    for leg in LEG_NAMES:
        for joint in ("trochanter", "tibia"):
            for direction in ("flexor", "extensor"):
                ids = (next_id,)
                next_id += 1
                groups.append(
                    MotorGroup(
                        name=f"{leg}_{joint}_{direction}",
                        leg=leg,
                        joint=joint,
                        direction=direction,
                        neuron_ids=ids,
                    )
                )
    motor = HexapodMotorMap(groups=tuple(groups))
    ids = np.array(
        [1, 2, 3, 4, 5, 6, *(group.neuron_ids[0] for group in groups)],
        dtype=np.uint64,
    )
    index = {int(value): position for position, value in enumerate(ids)}
    rows = []
    columns = []
    values = []
    if connected:
        for dn_id, group in zip(range(1, 7), groups[:6], strict=True):
            rows.append(index[dn_id])
            columns.append(index[group.neuron_ids[0]])
            values.append(120.0)
    outgoing = csr_array(
        (np.array(values, dtype=np.float32), (rows, columns)),
        shape=(len(ids), len(ids)),
        dtype=np.float32,
    )
    graph = EventConnectome(
        neuron_ids=ids,
        cell_types=("fixture",) * len(ids),
        roles=("fixture",) * len(ids),
        transmitters=("acetylcholine",) * len(ids),
        superclasses=("fixture",) * len(ids),
        outgoing=outgoing,
    )
    return graph, descending, motor


def test_connected_dn_to_motor_paths_are_positive_and_causal() -> None:
    graph, descending, motor = fixture(connected=True)

    result = run_dn_to_motor_protocols(
        graph,
        descending,
        motor,
        steps=90,
        seed=4,
    )

    assert all(claim.classification == "positive" for claim in result.claims.values())
    by_name = {condition.name: condition for condition in result.conditions}
    normal = by_name["d_na02_left:normal"]
    dn_lesion = by_name["d_na02_left:dn_lesion"]
    motor_lesion = by_name["d_na02_left:motor_lesion"]
    assert normal.motor_spikes > 0
    assert dn_lesion.motor_spikes == 0
    assert motor_lesion.motor_spikes == 0
    assert by_name["d_na02_left:restored"].trace_digest == normal.trace_digest
    assert by_name["d_na02_left:replay"].trace_digest == normal.trace_digest
    assert result.graph_unchanged
    assert result.replay_exact


def test_zero_edge_dn_to_motor_paths_remain_null() -> None:
    graph, descending, motor = fixture(connected=False)

    result = run_dn_to_motor_protocols(
        graph,
        descending,
        motor,
        steps=90,
        seed=4,
    )

    assert all(claim.classification == "null" for claim in result.claims.values())
    assert all(
        condition.motor_spikes == 0
        for condition in result.conditions
        if condition.name.endswith(":normal")
    )


def test_mirrored_dn_conditions_use_homologous_inputs() -> None:
    graph, descending, motor = fixture(connected=True)
    result = run_dn_to_motor_protocols(
        graph,
        descending,
        motor,
        steps=90,
        seed=4,
    )
    by_name = {condition.name: condition for condition in result.conditions}

    left = by_name["d_ng13_left:normal"]
    mirror = by_name["d_ng13_left:mirror"]

    assert left.input_population == "d_ng13_left"
    assert mirror.input_population == "d_ng13_right"
    assert left.motor_spikes == mirror.motor_spikes


def test_protocol_never_reports_direct_body_motion() -> None:
    graph, descending, motor = fixture(connected=True)
    result = run_dn_to_motor_protocols(
        graph,
        descending,
        motor,
        steps=90,
        seed=4,
    )

    forbidden = {"forward", "turn", "thrust", "yaw", "body", "joint_motion"}
    assert forbidden.isdisjoint(type(result.conditions[0]).model_fields)
