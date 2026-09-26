import numpy as np
from scipy.sparse import csr_array

from flybrain import hexapod_neural_protocols as neural_protocols
from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import LEG_NAMES
from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS, HexapodMotorMap, MotorGroup
from flybrain.hexapod_neural_protocols import run_dn_to_motor_protocols
from flybrain.proprioceptive_interface import ProprioceptiveBank, ProprioceptiveMap


def fixture(*, connected: bool) -> tuple[EventConnectome, DescendingMap, HexapodMotorMap]:
    descending = DescendingMap((1,), (2,), (3,), (4,), (5,), (6,))
    groups = []
    next_id = 100
    for name, leg, joint, direction in CANONICAL_MOTOR_GROUPS:
        ids = (next_id,)
        next_id += 1
        groups.append(
            MotorGroup(
                name=name,
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


def proprio_fixture(
    *, connected: bool, target_joint: str = "trochanter"
) -> tuple[EventConnectome, ProprioceptiveMap, HexapodMotorMap]:
    banks = tuple(
        ProprioceptiveBank(
            name=f"{leg}_proprioception",
            leg=leg,
            neuron_ids=(10 + index,),
        )
        for index, leg in enumerate(LEG_NAMES)
    )
    proprio = ProprioceptiveMap(banks=banks)
    groups = []
    next_id = 100
    for name, leg, joint, direction in CANONICAL_MOTOR_GROUPS:
        groups.append(
            MotorGroup(
                name=name,
                leg=leg,
                joint=joint,
                direction=direction,
                neuron_ids=(next_id,),
            )
        )
        next_id += 1
    motor = HexapodMotorMap(groups=tuple(groups))
    ids = np.array(
        [
            *(bank.neuron_ids[0] for bank in banks),
            *(group.neuron_ids[0] for group in groups),
        ],
        dtype=np.uint64,
    )
    index_by_id = {int(value): position for position, value in enumerate(ids)}
    rows = []
    columns = []
    values = []
    if connected:
        for bank, leg in zip(banks, LEG_NAMES, strict=True):
            group = motor.group(f"{leg}_{target_joint}_flexor")
            rows.append(index_by_id[bank.neuron_ids[0]])
            columns.append(index_by_id[group.neuron_ids[0]])
            values.append(500.0)
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
    return graph, proprio, motor


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


def test_connected_proprio_to_motor_paths_are_positive_and_causal() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)

    result = neural_protocols.run_proprio_to_motor_protocols(
        graph,
        proprio,
        motor,
        steps=90,
        seed=8,
    )

    assert all(claim.classification == "positive" for claim in result.claims.values())
    by_name = {condition.name: condition for condition in result.conditions}
    normal = by_name["left_fore_proprioception:normal"]
    assert normal.event_target_ids == (10,)
    assert normal.input_spikes_by_bank == {
        "left_fore_proprioception": normal.input_spikes,
        "right_fore_proprioception": 0,
        "left_middle_proprioception": 0,
        "right_middle_proprioception": 0,
        "left_hind_proprioception": 0,
        "right_hind_proprioception": 0,
    }
    assert normal.motor_spikes > 0
    assert by_name["left_fore_proprioception:side_lesion"].motor_spikes == 0
    assert by_name["left_fore_proprioception:motor_lesion"].motor_spikes == 0
    assert by_name["left_fore_proprioception:restored"].trace_digest == normal.trace_digest
    assert by_name["left_fore_proprioception:replay"].trace_digest == normal.trace_digest
    assert result.replay_exact
    assert result.graph_unchanged


def test_zero_edge_proprio_to_motor_paths_remain_null() -> None:
    graph, proprio, motor = proprio_fixture(connected=False)

    result = neural_protocols.run_proprio_to_motor_protocols(
        graph,
        proprio,
        motor,
        steps=90,
        seed=8,
    )

    assert all(claim.classification == "null" for claim in result.claims.values())
    assert all(
        condition.motor_spikes == 0
        for condition in result.conditions
        if condition.name.endswith(":normal")
    )


def test_proprio_mirror_uses_only_homologous_bank() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)
    result = neural_protocols.run_proprio_to_motor_protocols(
        graph,
        proprio,
        motor,
        steps=90,
        seed=8,
    )
    by_name = {condition.name: condition for condition in result.conditions}

    left = by_name["left_middle_proprioception:normal"]
    mirror = by_name["left_middle_proprioception:mirror"]

    assert left.input_bank == "left_middle_proprioception"
    assert mirror.input_bank == "right_middle_proprioception"
    assert mirror.event_target_ids == (13,)
    assert left.motor_spikes == mirror.motor_spikes


def test_proprio_protocol_exposes_no_privileged_state_or_body_command() -> None:
    forbidden = {
        "position",
        "target",
        "food",
        "threat",
        "reward",
        "desired_action",
        "forward",
        "turn",
        "thrust",
        "yaw",
        "body",
    }

    assert forbidden.isdisjoint(
        neural_protocols.ProprioceptiveConditionResult.model_fields
    )


def test_closed_loop_preserves_neural_state_and_routes_body_feedback() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)

    result = neural_protocols.run_closed_loop_hexapod_protocol(
        graph,
        proprio,
        motor,
        body_steps=6,
        seed=12,
    )

    by_name = {condition.name: condition for condition in result.conditions}
    normal = by_name["normal"]
    assert normal.neural_chunk_steps == 10
    assert normal.neural_state_steps == (10, 20, 30, 40, 50, 60)
    assert normal.proprioceptive_event_steps == (0, 10, 20, 30, 40, 50)
    assert normal.motor_spikes > 0
    assert normal.neural_joint_motion_rad > 0.0
    assert normal.final_body.time_s == 0.006
    assert result.graph_unchanged
    assert result.sparse_storage_unchanged


def test_closed_loop_lesions_restore_and_replay_exactly() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)

    result = neural_protocols.run_closed_loop_hexapod_protocol(
        graph,
        proprio,
        motor,
        body_steps=6,
        seed=12,
    )

    by_name = {condition.name: condition for condition in result.conditions}
    normal = by_name["normal"]
    assert by_name["motor_lesion"].motor_spikes == 0
    assert by_name["motor_lesion"].neural_joint_motion_rad == 0.0
    assert by_name["proprio_lesion"].motor_spikes == 0
    assert by_name["proprio_lesion"].neural_joint_motion_rad == 0.0
    assert by_name["restored"].trace_digest == normal.trace_digest
    assert by_name["replay"].trace_digest == normal.trace_digest
    assert result.mirror_exact
    assert result.replay_exact
    assert result.mirror_exact
    assert by_name["normal"].proprioceptive_input_mapping == {
        "left_fore": "left_fore_proprioception",
        "right_fore": "right_fore_proprioception",
        "left_middle": "left_middle_proprioception",
        "right_middle": "right_middle_proprioception",
        "left_hind": "left_hind_proprioception",
        "right_hind": "right_hind_proprioception",
    }
    assert by_name["mirror"].proprioceptive_input_mapping == {
        "left_fore": "right_fore_proprioception",
        "right_fore": "left_fore_proprioception",
        "left_middle": "right_middle_proprioception",
        "right_middle": "left_middle_proprioception",
        "left_hind": "right_hind_proprioception",
        "right_hind": "left_hind_proprioception",
    }


def test_closed_loop_contract_has_no_target_reward_or_desired_action() -> None:
    forbidden = {
        "target",
        "food",
        "threat",
        "reward",
        "desired_action",
        "desired_heading",
    }

    assert forbidden.isdisjoint(
        neural_protocols.ClosedLoopConditionResult.model_fields
    )
    assert forbidden.isdisjoint(
        neural_protocols.ClosedLoopHexapodResult.model_fields
    )


def test_closed_loop_classifies_connected_and_zero_edge_graphs_independently() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)
    connected = neural_protocols.run_closed_loop_hexapod_protocol(
        graph,
        proprio,
        motor,
        body_steps=6,
        seed=12,
    )
    zero_graph, zero_proprio, zero_motor = proprio_fixture(connected=False)
    zero = neural_protocols.run_closed_loop_hexapod_protocol(
        zero_graph,
        zero_proprio,
        zero_motor,
        body_steps=6,
        seed=12,
    )

    connected_normal = connected.conditions[0]
    zero_normal = zero.conditions[0]
    assert connected_normal.proprioceptive_spikes > 0
    assert zero_normal.proprioceptive_spikes > 0
    assert connected.claim.classification == "positive"
    assert zero.claim.classification == "null"
