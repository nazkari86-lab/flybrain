import math

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.hexapod_benchmark import (
    HexapodBenchmarkThresholds,
    ProtocolName,
    _build_phase_gait_schedule,
    build_direct_motor_schedules,
    run_hexapod_benchmark,
)
from flybrain.hexapod_body import HexapodParameters, ReferenceHexapod
from flybrain.hexapod_motor import (
    CANONICAL_MOTOR_GROUPS,
    HexapodMotorDecoder,
    HexapodMotorMap,
    MotorGroup,
)


def fixture_map_and_graph() -> tuple[HexapodMotorMap, EventConnectome]:
    groups = []
    next_id = 1
    for name, leg, joint, direction in CANONICAL_MOTOR_GROUPS:
        size = 2 + (direction in {"flexor", "anterior"}) + leg.startswith("right")
        ids = tuple(range(next_id, next_id + size))
        next_id += size
        groups.append(
            MotorGroup(
                name=name,
                leg=leg,
                joint=joint,
                direction=direction,
                neuron_ids=ids,
            )
        )
    mapping = HexapodMotorMap(groups=tuple(groups))
    ids = np.array(
        sorted(neuron_id for group in mapping.groups for neuron_id in group.neuron_ids),
        dtype=np.uint64,
    )
    size = len(ids)
    graph = EventConnectome(
        neuron_ids=ids,
        cell_types=("motor",) * size,
        roles=("motor",) * size,
        transmitters=("acetylcholine",) * size,
        superclasses=("vnc_motor",) * size,
        outgoing=csr_array((size, size), dtype=np.float32),
    )
    return mapping, graph


@pytest.fixture(scope="module")
def benchmark_result():
    mapping, graph = fixture_map_and_graph()
    return run_hexapod_benchmark(
        graph,
        mapping,
        steps=24,
        thresholds=HexapodBenchmarkThresholds(),
    )


def test_all_direct_schedules_are_prebuilt_hashed_and_replayable() -> None:
    mapping, _ = fixture_map_and_graph()

    first = build_direct_motor_schedules(mapping, steps=24)
    second = build_direct_motor_schedules(mapping, steps=24)

    assert first == second
    assert len(first) == 36
    assert len({item.sha256 for item in first}) == 36
    assert all(len(item.spikes_by_step) == 24 for item in first)


def test_every_motor_group_has_separate_causal_classification(benchmark_result) -> None:
    mapping, _ = fixture_map_and_graph()

    assert set(benchmark_result.direct_motor_claims) == {
        group.name for group in mapping.groups
    }
    assert all(
        claim.classification == "positive"
        for claim in benchmark_result.direct_motor_claims.values()
    )


def test_flexor_and_extensor_conditions_have_opposite_joint_motion(
    benchmark_result,
) -> None:
    by_name = {condition.name: condition for condition in benchmark_result.conditions}
    flexor = by_name["direct:left_fore:tibia:flexor:normal"]
    extensor = by_name["direct:left_fore:tibia:extensor:normal"]

    assert flexor.joint_motion_rad is not None
    assert extensor.joint_motion_rad is not None
    assert flexor.joint_motion_rad < 0.0
    assert extensor.joint_motion_rad > 0.0


def test_matching_lesion_removes_motion_without_opposite_injection(
    benchmark_result,
) -> None:
    by_name = {condition.name: condition for condition in benchmark_result.conditions}
    normal = by_name["direct:right_middle:trochanter:extensor:normal"]
    lesion = by_name["direct:right_middle:trochanter:extensor:matching_lesion"]
    opposite = by_name["direct:right_middle:trochanter:extensor:opposite_lesion"]

    assert normal.joint_motion_rad is not None
    assert lesion.joint_motion_rad == 0.0
    assert lesion.motor_spikes == 0
    assert opposite.trace_digest == normal.trace_digest


def test_restoration_replay_mirror_and_holdout_preserve_direction(
    benchmark_result,
) -> None:
    by_name = {condition.name: condition for condition in benchmark_result.conditions}
    normal = by_name["direct:left_hind:tibia:flexor:normal"]
    restored = by_name["direct:left_hind:tibia:flexor:restored"]
    replay = by_name["direct:left_hind:tibia:flexor:replay"]
    mirror = by_name["direct:left_hind:tibia:flexor:mirror"]
    perturbation = by_name["direct:left_hind:tibia:flexor:perturbation"]
    holdout = by_name["direct:left_hind:tibia:flexor:holdout"]

    assert restored.trace_digest == normal.trace_digest
    assert replay.trace_digest == normal.trace_digest
    assert mirror.joint_motion_rad is not None and mirror.joint_motion_rad < 0.0
    assert perturbation.joint_motion_rad is not None and perturbation.joint_motion_rad < 0.0
    assert holdout.joint_motion_rad is not None and holdout.joint_motion_rad < 0.0


def test_phase_gait_is_classified_without_forcing_a_positive_result(
    benchmark_result,
) -> None:
    claim = benchmark_result.gait_claim

    assert claim.classification in {
        "positive",
        "null",
        "directionally_wrong",
        "underpowered",
    }
    assert claim.evidence_kind == "simulation_observation"
    gait_conditions = [
        condition
        for condition in benchmark_result.conditions
        if condition.protocol == ProtocolName.PHASE_GAIT_CALIBRATION
    ]
    assert gait_conditions
    assert all(math.isfinite(item.energy_j) for item in gait_conditions)


def test_phase_schedule_preserves_support_while_generating_forward_propulsion() -> None:
    mapping, _ = fixture_map_and_graph()
    parameters = HexapodParameters()
    schedule = _build_phase_gait_schedule(mapping, parameters, steps=180)
    body = ReferenceHexapod(parameters)
    decoder = HexapodMotorDecoder(mapping)
    minimum_support = body.observe().support_count

    for scheduled in schedule.spikes_by_step:
        current = body.observe()
        state = decoder.decode(
            scheduled,
            window_s=parameters.dt_s,
            leg_phases=tuple(leg.phase for leg in current.legs),
        )
        body.step(state.torques)
        minimum_support = min(minimum_support, body.observe().support_count)

    assert minimum_support >= 3
    assert not body.observe().fallen
    assert body.observe().thorax_position_m[0] > 1e-5


def test_phase_schedule_uses_anatomical_antagonists_on_both_sides() -> None:
    mapping, _ = fixture_map_and_graph()
    first = set(
        _build_phase_gait_schedule(mapping, HexapodParameters(), steps=1)
        .spikes_by_step[0]
    )

    for leg in ("left_fore", "right_middle", "left_hind"):
        anterior = mapping.group(f"{leg}_thorax_coxa_anterior")
        assert anterior.neuron_ids[0] in first
        assert set(mapping.group(f"{leg}_thorax_coxa_posterior").neuron_ids).isdisjoint(first)
    for leg in ("right_fore", "left_middle", "right_hind"):
        posterior = mapping.group(f"{leg}_thorax_coxa_posterior")
        assert set(posterior.neuron_ids).issubset(first)
        assert set(mapping.group(f"{leg}_thorax_coxa_anterior").neuron_ids).isdisjoint(first)


def test_integrity_replay_bounds_and_energy_gates_pass(benchmark_result) -> None:
    assert benchmark_result.graph_unchanged
    assert benchmark_result.replay_exact
    assert benchmark_result.calibration_gates["joint_bounds"]
    assert benchmark_result.calibration_gates["finite_state"]
    assert benchmark_result.calibration_gates["nonnegative_energy"]
    assert benchmark_result.calibration_passed


def test_zero_edge_graph_still_labels_direct_drive_as_calibration_only(
    benchmark_result,
) -> None:
    assert benchmark_result.graph_edges == 0
    assert all(
        condition.protocol
        in {ProtocolName.MOTOR_GROUP_CALIBRATION, ProtocolName.PHASE_GAIT_CALIBRATION}
        for condition in benchmark_result.conditions
    )
    assert not hasattr(benchmark_result, "dn_to_motor_claim")
