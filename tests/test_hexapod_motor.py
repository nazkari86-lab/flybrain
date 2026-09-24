import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.sparse import csr_array

from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import LEG_NAMES
from flybrain.hexapod_motor import (
    CANONICAL_MOTOR_GROUPS,
    HexapodMotorDecoder,
    HexapodMotorMap,
    MotorActivationState,
    MotorGroup,
    PhaseEnvelopeAssumption,
    motor_population_silence_mask,
)

RETAINED_MALE_CNS = Path(
    "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5"
)


def test_canonical_motor_map_includes_neural_thorax_coxa_antagonists() -> None:
    assert len(CANONICAL_MOTOR_GROUPS) == 36
    names = {name for name, *_ in CANONICAL_MOTOR_GROUPS}
    for leg in LEG_NAMES:
        assert f"{leg}_thorax_coxa_anterior" in names
        assert f"{leg}_thorax_coxa_posterior" in names


def motor_map(*, overlap: bool = False) -> HexapodMotorMap:
    groups = []
    next_id = 1
    for leg in LEG_NAMES:
        for joint, directions in (
            ("thorax_coxa", ("anterior", "posterior")),
            ("trochanter", ("flexor", "extensor")),
            ("tibia", ("flexor", "extensor")),
        ):
            for direction in directions:
                size = 1 if direction == "extensor" else 2 + leg.startswith("right")
                ids = tuple(range(next_id, next_id + size))
                next_id += size
                groups.append(
                    MotorGroup(
                        name=f"{leg}_{joint}_{direction}",
                        leg=leg,
                        joint=joint,
                        direction=direction,
                        neuron_ids=ids,
                    )
                )
    if overlap:
        groups[-1] = groups[-1].model_copy(
            update={"neuron_ids": groups[0].neuron_ids}
        )
    return HexapodMotorMap(groups=tuple(groups))


def graph_for(mapping: HexapodMotorMap, *, omit_last: bool = False) -> EventConnectome:
    neuron_ids = sorted(
        neuron_id for group in mapping.groups for neuron_id in group.neuron_ids
    )
    if omit_last:
        neuron_ids.pop()
    size = len(neuron_ids)
    return EventConnectome(
        neuron_ids=np.array(neuron_ids, dtype=np.uint64),
        cell_types=("motor",) * size,
        roles=("motor",) * size,
        transmitters=("acetylcholine",) * size,
        superclasses=("vnc_motor",) * size,
        outgoing=csr_array((size, size), dtype=np.float32),
    )


def test_motor_map_requires_complete_disjoint_positive_populations() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        motor_map(overlap=True)

    mapping = motor_map()
    with pytest.raises(ValidationError, match="positive"):
        MotorGroup.model_validate(
            {**mapping.groups[0].model_dump(), "neuron_ids": (0,)}
        )
    with pytest.raises(ValidationError, match="complete"):
        HexapodMotorMap(groups=mapping.groups[:-1])


def test_motor_map_builds_only_from_resolved_real_registry() -> None:
    if not RETAINED_MALE_CNS.is_dir():
        pytest.skip(f"retained snapshot unavailable: {RETAINED_MALE_CNS}")
    registry = resolve_biological_registry(
        load_biological_registry(
            Path("data/registry/hexapod-motor-registry-v2.json")
        ),
        RETAINED_MALE_CNS,
    )

    mapping = HexapodMotorMap.from_registry(registry)

    assert len(mapping.groups) == 36
    assert sum(len(group.neuron_ids) for group in mapping.groups) == 182


def test_population_size_normalizes_rate_and_not_motor_authority() -> None:
    mapping = motor_map()
    decoder = HexapodMotorDecoder(mapping, saturation_rate_hz=100.0)
    left = mapping.group("left_fore_trochanter_flexor")
    right = mapping.group("right_fore_trochanter_flexor")
    spikes = (*left.neuron_ids, *right.neuron_ids)

    state = decoder.decode(spikes, window_s=0.01, leg_phases=(0.0,) * 6)

    assert state.rate_hz(left.name) == pytest.approx(100.0)
    assert state.rate_hz(right.name) == pytest.approx(100.0)
    assert state.activation(left.name) == pytest.approx(state.activation(right.name))
    assert state.torques.values[0][1] == pytest.approx(state.torques.values[1][1])


def test_flexor_and_extensor_spikes_produce_opposite_joint_signs() -> None:
    mapping = motor_map()
    flexor = mapping.group("left_middle_tibia_flexor")
    extensor = mapping.group("left_middle_tibia_extensor")
    flexed = HexapodMotorDecoder(mapping).decode(
        flexor.neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )
    extended = HexapodMotorDecoder(mapping).decode(
        extensor.neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )

    assert flexed.torques.values[2][2] < 0.0
    assert extended.torques.values[2][2] > 0.0
    assert flexed.torques.values[2][2] == pytest.approx(
        -extended.torques.values[2][2]
    )


def test_anterior_and_posterior_rotator_spikes_drive_thorax_coxa() -> None:
    mapping = motor_map()
    anterior = mapping.group("left_fore_thorax_coxa_anterior")
    posterior = mapping.group("left_fore_thorax_coxa_posterior")
    advanced = HexapodMotorDecoder(mapping).decode(
        anterior.neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )
    retracted = HexapodMotorDecoder(mapping).decode(
        posterior.neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )

    assert advanced.torques.values[0][0] > 0.0
    assert retracted.torques.values[0][0] < 0.0
    assert advanced.torques.values[0][0] == pytest.approx(
        -retracted.torques.values[0][0]
    )


def test_homologous_thorax_coxa_rotators_produce_mirrored_joint_torque() -> None:
    mapping = motor_map()
    left = HexapodMotorDecoder(mapping).decode(
        mapping.group("left_fore_thorax_coxa_anterior").neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )
    right = HexapodMotorDecoder(mapping).decode(
        mapping.group("right_fore_thorax_coxa_anterior").neuron_ids,
        window_s=0.01,
        leg_phases=(0.0,) * 6,
    )

    assert left.torques.values[0][0] > 0.0
    assert right.torques.values[1][0] < 0.0
    assert left.torques.values[0][0] == pytest.approx(-right.torques.values[1][0])


def test_activation_is_bounded_low_pass_and_replayable() -> None:
    mapping = motor_map()
    group = mapping.group("left_hind_tibia_extensor")
    first = HexapodMotorDecoder(mapping, activation_time_constant_s=0.02)
    second = HexapodMotorDecoder(mapping, activation_time_constant_s=0.02)
    schedule = (group.neuron_ids, (), group.neuron_ids * 4, ())

    first_trace = tuple(
        first.decode(spikes, window_s=0.01, leg_phases=(0.25,) * 6)
        for spikes in schedule
    )
    second_trace = tuple(
        second.decode(spikes, window_s=0.01, leg_phases=(0.25,) * 6)
        for spikes in schedule
    )

    assert first_trace == second_trace
    assert all(
        0.0 <= activation <= 1.0
        for state in first_trace
        for activation in state.activations
    )
    assert first_trace[0].activation(group.name) < 1.0
    assert first_trace[1].activation(group.name) < first_trace[0].activation(group.name)


def test_phase_envelope_is_serialized_and_modulates_only_thorax_coxa() -> None:
    mapping = motor_map()
    assumption = PhaseEnvelopeAssumption(amplitude_nm=0.004)
    decoder = HexapodMotorDecoder(mapping, phase_envelope=assumption)

    state = decoder.decode((), window_s=0.01, leg_phases=(0.25, 0.75) * 3)

    assert state.phase_envelope == assumption
    assert all(torque[1:] == (0.0, 0.0) for torque in state.torques.values)
    assert state.torques.values[0][0] == pytest.approx(0.004)
    assert state.torques.values[1][0] == pytest.approx(-0.004)


def test_declared_descending_spikes_modulate_phase_without_target_commands() -> None:
    mapping = motor_map()
    descending = DescendingMap(
        d_na02_left=(1001,),
        d_na02_right=(1002,),
        d_ng13_left=(1003,),
        d_ng13_right=(1004,),
        mdn_left=(1005,),
        mdn_right=(1006,),
    )
    left_decoder = HexapodMotorDecoder(
        mapping,
        phase_envelope=PhaseEnvelopeAssumption(amplitude_nm=0.004),
        descending_map=descending,
    )
    right_decoder = HexapodMotorDecoder(
        mapping,
        phase_envelope=PhaseEnvelopeAssumption(amplitude_nm=0.004),
        descending_map=descending,
    )

    left = left_decoder.decode(
        (1001,), window_s=0.01, leg_phases=(0.125,) * 6
    )
    right = right_decoder.decode(
        (1002,), window_s=0.01, leg_phases=(0.125,) * 6
    )

    assert left.descending_phase_bias > 0.0
    assert right.descending_phase_bias < 0.0
    assert left.torques.values[0][0] > right.torques.values[0][0]


def test_descending_spikes_modulate_neural_torque_without_external_phase_drive() -> None:
    mapping = motor_map()
    descending = DescendingMap(
        d_na02_left=(1001,),
        d_na02_right=(1002,),
        d_ng13_left=(1003,),
        d_ng13_right=(1004,),
        mdn_left=(1005,),
        mdn_right=(1006,),
    )
    decoder = HexapodMotorDecoder(mapping, descending_map=descending)
    anterior = mapping.group("left_fore_thorax_coxa_anterior")
    motor_only = decoder.decode(
        anterior.neuron_ids, window_s=0.01, leg_phases=(0.125,) * 6
    )
    decoder = HexapodMotorDecoder(mapping, descending_map=descending)
    motor_with_mdn = decoder.decode(
        (*anterior.neuron_ids, 1005), window_s=0.01, leg_phases=(0.125,) * 6
    )

    assert motor_with_mdn.torques.values[0][0] != pytest.approx(
        motor_only.torques.values[0][0]
    )


def test_lateral_descending_spikes_modulate_existing_neural_torque_without_phase_drive() -> None:
    mapping = motor_map()
    descending = DescendingMap(
        d_na02_left=(1001,), d_na02_right=(1002,),
        d_ng13_left=(1003,), d_ng13_right=(1004,),
        mdn_left=(1005,), mdn_right=(1006,),
    )
    motor_id = mapping.group("left_fore_thorax_coxa_anterior").neuron_ids[0]

    def decode(spikes):
        return HexapodMotorDecoder(mapping, descending_map=descending).decode(
            spikes, window_s=0.01, leg_phases=(0.0,) * 6
        )

    motor_only = decode((motor_id,))
    left_drive = decode((motor_id, 1001, 1003))
    right_drive = decode((motor_id, 1002, 1004))
    na_left = decode((motor_id, 1001))
    ng_left = decode((motor_id, 1003))
    dn_only = decode((1001, 1003))

    assert left_drive.phase_envelope.amplitude_nm == 0.0
    assert left_drive.torques.values[0][0] > motor_only.torques.values[0][0]
    assert right_drive.torques.values[0][0] < motor_only.torques.values[0][0]
    assert na_left.torques.values[0][0] > motor_only.torques.values[0][0]
    assert ng_left.torques.values[0][0] > motor_only.torques.values[0][0]
    assert dn_only.torques.values == ((0.0, 0.0, 0.0),) * 6


def test_descending_phase_modulation_uses_rate_not_raw_spike_count() -> None:
    descending = DescendingMap(
        d_na02_left=(1001,), d_na02_right=(1002,),
        d_ng13_left=(1003,), d_ng13_right=(1004,),
        mdn_left=(1005,), mdn_right=(1006,),
    )
    decoder = HexapodMotorDecoder(motor_map(), descending_map=descending)

    state = decoder.decode(
        (1001,), window_s=0.02, leg_phases=(0.0,) * 6
    )

    assert state.descending_phase_bias == pytest.approx(0.125)


def test_motor_state_has_no_direct_body_command_or_privileged_fields() -> None:
    forbidden = {
        "forward",
        "turn",
        "yaw",
        "thrust",
        "target",
        "food",
        "threat",
        "reward",
        "desired_action",
    }

    assert forbidden.isdisjoint(MotorActivationState.model_fields)


def test_named_motor_silence_mask_is_exact_and_unknown_names_fail() -> None:
    mapping = motor_map()
    graph = graph_for(mapping)
    selected = mapping.group("right_hind_tibia_flexor")

    mask = motor_population_silence_mask(graph, mapping, frozenset({selected.name}))

    assert int(mask.sum()) == len(selected.neuron_ids)
    masked_ids = set(int(value) for value in graph.neuron_ids[mask])
    assert masked_ids == set(selected.neuron_ids)
    with pytest.raises(ValueError, match="unknown motor populations"):
        motor_population_silence_mask(graph, mapping, frozenset({"unknown"}))
    with pytest.raises(ValueError, match="absent from graph"):
        motor_population_silence_mask(
            graph_for(mapping, omit_last=True), mapping, frozenset()
        )


def test_silencing_removes_drive_without_injecting_opposite_activation() -> None:
    mapping = motor_map()
    selected = mapping.group("left_fore_trochanter_extensor")
    graph = graph_for(mapping)
    mask = motor_population_silence_mask(graph, mapping, frozenset({selected.name}))
    delivered = tuple(
        int(neuron_id)
        for neuron_id, silenced in zip(graph.neuron_ids, mask, strict=True)
        if neuron_id in selected.neuron_ids and not silenced
    )

    state = HexapodMotorDecoder(mapping).decode(
        delivered, window_s=0.01, leg_phases=(0.0,) * 6
    )

    assert state.activation(selected.name) == 0.0
    assert state.torques.values[0][1] == 0.0
    assert all(math.isfinite(value) for torque in state.torques.values for value in torque)
