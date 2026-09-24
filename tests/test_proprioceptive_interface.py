from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.sparse import csr_array

from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import (
    LEG_NAMES,
    HexapodParameters,
    HexapodTorque,
    ReferenceHexapod,
)
from flybrain.proprioceptive_interface import (
    ProprioceptiveBank,
    ProprioceptiveCalibration,
    ProprioceptiveEncoder,
    ProprioceptiveMap,
    ProprioceptiveObservation,
    observe_proprioception,
)

RETAINED_MALE_CNS = Path(
    "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5"
)


def proprioceptive_map(*, overlap: bool = False) -> ProprioceptiveMap:
    banks = []
    next_id = 1
    for index, leg in enumerate(LEG_NAMES):
        size = index + 1
        ids = tuple(range(next_id, next_id + size))
        next_id += size
        banks.append(
            ProprioceptiveBank(
                name=f"{leg}_proprioception",
                leg=leg,
                neuron_ids=ids,
            )
        )
    if overlap:
        banks[-1] = ProprioceptiveBank(
            name="right_hind_proprioception",
            leg="right_hind",
            neuron_ids=banks[0].neuron_ids,
        )
    return ProprioceptiveMap(banks=tuple(banks))


def body_with_leg_values(
    *,
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact: bool = False,
    load: float = 0.0,
):
    body = ReferenceHexapod().observe()
    leg = body.leg("left_fore").model_copy(
        update={
            "joint_angles_rad": angles,
            "joint_velocities_rad_s": velocities,
            "contact": contact,
            "load_n": load,
        }
    )
    legs = (leg, *body.legs[1:])
    return body.model_copy(update={"legs": legs})


def graph_for(mapping: ProprioceptiveMap, *, omit_last: bool = False) -> EventConnectome:
    ids = sorted(neuron_id for bank in mapping.banks for neuron_id in bank.neuron_ids)
    if omit_last:
        ids.pop()
    size = len(ids)
    return EventConnectome(
        neuron_ids=np.array(ids, dtype=np.uint64),
        cell_types=("sensory",) * size,
        roles=("sensory",) * size,
        transmitters=("acetylcholine",) * size,
        superclasses=("vnc_sensory",) * size,
        outgoing=csr_array((size, size), dtype=np.float32),
    )


def test_observation_normalizes_angles_velocities_contact_and_load_monotonically() -> None:
    parameters = HexapodParameters()
    calibration = ProprioceptiveCalibration(
        velocity_scale_rad_s=2.0,
        load_scale_n=0.5,
    )
    low = observe_proprioception(
        body_with_leg_values(
            angles=parameters.joint_lower_rad,
            velocities=(-2.0, -2.0, -2.0),
            contact=False,
            load=0.0,
        ),
        parameters,
        calibration,
    ).leg("left_fore")
    high = observe_proprioception(
        body_with_leg_values(
            angles=parameters.joint_upper_rad,
            velocities=(2.0, 2.0, 2.0),
            contact=True,
            load=0.5,
        ),
        parameters,
        calibration,
    ).leg("left_fore")

    assert low.joint_angles == (0.0, 0.0, 0.0)
    assert high.joint_angles == (1.0, 1.0, 1.0)
    assert low.joint_velocities == (0.0, 0.0, 0.0)
    assert high.joint_velocities == (1.0, 1.0, 1.0)
    assert low.contact == 0.0
    assert high.contact == 1.0
    assert low.load == 0.0
    assert high.load == 1.0


def test_observation_contains_no_privileged_world_or_action_fields() -> None:
    forbidden = {
        "position",
        "target",
        "food",
        "threat",
        "reward",
        "desired_action",
        "forward",
        "turn",
    }
    observation = observe_proprioception(
        ReferenceHexapod().observe(),
        HexapodParameters(),
        ProprioceptiveCalibration(),
    )

    assert forbidden.isdisjoint(ProprioceptiveObservation.model_fields)
    assert all(forbidden.isdisjoint(type(leg).model_fields) for leg in observation.legs)


def test_mirror_swaps_homologous_observations_with_declared_joint_sign() -> None:
    simulator = ReferenceHexapod()
    body = simulator.step(
        HexapodTorque.for_leg("left_fore", (0.01, 0.02, -0.01))
    )
    parameters = simulator.parameters
    calibration = ProprioceptiveCalibration()
    normal = observe_proprioception(body, parameters, calibration)
    mirrored = observe_proprioception(body.mirror(), parameters, calibration)
    left = normal.leg("left_fore")
    right = mirrored.leg("right_fore")

    assert right.joint_angles[0] == pytest.approx(1.0 - left.joint_angles[0])
    assert right.joint_angles[1:] == pytest.approx(left.joint_angles[1:])
    assert right.joint_velocities[0] == pytest.approx(
        1.0 - left.joint_velocities[0]
    )
    assert right.joint_velocities[1:] == pytest.approx(left.joint_velocities[1:])
    assert right.contact == left.contact
    assert right.load == left.load


def test_encoding_normalizes_total_voltage_by_bank_size_and_replays_exactly() -> None:
    mapping = proprioceptive_map()
    calibration = ProprioceptiveCalibration(total_voltage=80.0)
    observation = observe_proprioception(
        body_with_leg_values(contact=True, load=calibration.load_scale_n),
        HexapodParameters(),
        calibration,
    )
    encoder = ProprioceptiveEncoder(mapping, calibration)

    first = encoder.encode(observation, step=7)
    second = encoder.encode(observation, step=7)

    assert first == second
    by_channel = {event.channel: event for event in first}
    left_fore = by_channel["proprioception_left_fore"]
    right_fore = by_channel["proprioception_right_fore"]
    right_hind = by_channel["proprioception_right_hind"]
    left_drive = sum(left_fore.voltages)
    right_drive = sum(right_hind.voltages)
    assert left_drive <= calibration.total_voltage
    assert right_drive <= calibration.total_voltage
    assert sum(right_fore.voltages) == pytest.approx(sum(right_hind.voltages))
    for event in first:
        assert len(set(event.voltages)) == 1
        assert sum(event.voltages) == pytest.approx(
            calibration.total_voltage
            * observation.leg(event.channel.removeprefix("proprioception_")).drive
        )


def test_source_equivalent_spikes_use_sparse_full_voltage_events_and_replay() -> None:
    mapping = proprioceptive_map()
    calibration = ProprioceptiveCalibration(total_voltage=80.0)
    observation = observe_proprioception(
        body_with_leg_values(contact=True, load=calibration.load_scale_n),
        HexapodParameters(),
        calibration,
    )
    encoder = ProprioceptiveEncoder(mapping, calibration)

    first = encoder.encode_source_equivalent_spikes(
        observation,
        steps=4,
        seed=7,
        rate_hz=10_000.0,
        dt_ms=0.1,
    )
    second = encoder.encode_source_equivalent_spikes(
        observation,
        steps=4,
        seed=7,
        rate_hz=10_000.0,
        dt_ms=0.1,
    )

    assert first == second
    assert first
    assert {event.channel for event in first} == {
        f"proprioception_spikes_{leg}" for leg in LEG_NAMES
    }
    assert all(
        set(event.voltages) == {calibration.total_voltage}
        for event in first
    )


def test_map_rejects_overlap_incomplete_and_missing_graph_ids() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        proprioceptive_map(overlap=True)
    mapping = proprioceptive_map()
    with pytest.raises(ValidationError, match="complete"):
        ProprioceptiveMap(banks=mapping.banks[:-1])
    with pytest.raises(ValueError, match="absent from graph"):
        mapping.validate_graph(graph_for(mapping, omit_last=True))


def test_encoder_rejects_invalid_steps_and_observation_inventory() -> None:
    mapping = proprioceptive_map()
    calibration = ProprioceptiveCalibration()
    encoder = ProprioceptiveEncoder(mapping, calibration)
    observation = observe_proprioception(
        ReferenceHexapod().observe(), HexapodParameters(), calibration
    )

    with pytest.raises(ValueError, match="step"):
        encoder.encode(observation, step=-1)
    with pytest.raises(ValueError, match="calibration"):
        encoder.encode(
            observation.model_copy(
                update={"calibration": ProprioceptiveCalibration(total_voltage=1.0)}
            ),
            step=0,
        )


def test_real_registry_builds_six_disjoint_proprioceptive_banks() -> None:
    if not RETAINED_MALE_CNS.is_dir():
        pytest.skip(f"retained snapshot unavailable: {RETAINED_MALE_CNS}")
    registry = resolve_biological_registry(
        load_biological_registry(
            Path("data/registry/hexapod-motor-registry-v2.json")
        ),
        RETAINED_MALE_CNS,
    )

    mapping = ProprioceptiveMap.from_registry(registry)

    assert len(mapping.banks) == 6
    assert sum(len(bank.neuron_ids) for bank in mapping.banks) == 622
