import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from flybrain.hexapod_body import (
    JOINT_NAMES,
    LEG_NAMES,
    HexapodBody,
    HexapodParameters,
    HexapodTorque,
    LegState,
    ReferenceHexapod,
)


def test_reference_body_has_six_ordered_three_joint_legs() -> None:
    simulator = ReferenceHexapod()
    body = simulator.observe()

    assert tuple(leg.name for leg in body.legs) == LEG_NAMES
    assert len(JOINT_NAMES) == 3
    assert all(len(leg.joint_angles_rad) == 3 for leg in body.legs)
    assert all(len(leg.joint_velocities_rad_s) == 3 for leg in body.legs)
    assert all(len(leg.applied_torques_nm) == 3 for leg in body.legs)
    assert body.support_count == 6
    assert body.support_margin_m > 0.0
    assert not body.fallen


@pytest.mark.parametrize(
    "mutation",
    [
        {"joint_angles_rad": (0.0, 0.0)},
        {"joint_velocities_rad_s": (0.0, math.nan, 0.0)},
        {"foot_position_m": (0.0, 0.0)},
        {"load_n": -1.0},
        {"phase": 1.0},
    ],
)
def test_leg_state_rejects_invalid_shape_or_nonfinite_values(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "name": "left_fore",
        "joint_angles_rad": (0.0, 0.0, 0.0),
        "joint_velocities_rad_s": (0.0, 0.0, 0.0),
        "applied_torques_nm": (0.0, 0.0, 0.0),
        "foot_position_m": (0.6, 0.5, 0.0),
        "contact": True,
        "load_n": 1.0,
        "phase": 0.0,
    }
    payload.update(mutation)

    with pytest.raises(ValidationError):
        LegState.model_validate(payload)


def test_body_and_parameter_records_are_deeply_immutable() -> None:
    simulator = ReferenceHexapod()
    body = simulator.observe()

    with pytest.raises(ValidationError):
        body.time_s = 2.0
    with pytest.raises(ValidationError):
        body.legs[0].load_n = 0.0
    with pytest.raises(ValidationError):
        simulator.parameters.max_torque_nm = (1.0, 1.0, 1.0)


def test_mirror_is_an_involution_and_swaps_homologous_legs() -> None:
    simulator = ReferenceHexapod()
    torque = HexapodTorque.for_leg("left_fore", (0.03, -0.02, 0.01))
    body = simulator.step(torque)

    mirrored = body.mirror()

    assert mirrored.leg("right_fore").joint_angles_rad[0] == pytest.approx(
        -body.leg("left_fore").joint_angles_rad[0]
    )
    assert mirrored.thorax_position_m[1] == pytest.approx(-body.thorax_position_m[1])
    assert mirrored.mirror() == body
    assert torque.mirror().mirror() == torque


def test_joint_limits_hold_under_sustained_excess_torque() -> None:
    parameters = HexapodParameters(dt_s=0.002)
    simulator = ReferenceHexapod(parameters)
    excessive = HexapodTorque(values=((100.0, -100.0, 100.0),) * 6)

    for _ in range(3000):
        body = simulator.step(excessive)

    for leg in body.legs:
        for value, lower, upper in zip(
            leg.joint_angles_rad,
            parameters.joint_lower_rad,
            parameters.joint_upper_rad,
            strict=True,
        ):
            assert lower <= value <= upper
        assert all(math.isfinite(value) for value in leg.joint_velocities_rad_s)


def test_energy_is_nonnegative_monotonic_and_zero_at_rest() -> None:
    rest = ReferenceHexapod()
    initial = rest.observe()
    for _ in range(20):
        rest_body = rest.step(HexapodTorque.zero())
    assert rest_body.energy_j == 0.0
    assert rest_body.thorax_position_m == initial.thorax_position_m
    assert rest_body.thorax_velocity_m_s == initial.thorax_velocity_m_s
    assert tuple(leg.joint_angles_rad for leg in rest_body.legs) == tuple(
        leg.joint_angles_rad for leg in initial.legs
    )
    assert tuple(leg.joint_velocities_rad_s for leg in rest_body.legs) == tuple(
        leg.joint_velocities_rad_s for leg in initial.legs
    )

    active = ReferenceHexapod()
    energies = []
    drive = HexapodTorque.for_leg("left_middle", (0.01, 0.02, -0.02))
    for _ in range(40):
        energies.append(active.step(drive).energy_j)
    assert energies == sorted(energies)
    assert energies[-1] > 0.0


def test_fixed_input_replay_is_bit_exact() -> None:
    first = ReferenceHexapod()
    second = ReferenceHexapod()
    schedule = (
        HexapodTorque.for_leg("left_fore", (0.01, 0.02, -0.01)),
        HexapodTorque.for_leg("right_middle", (-0.01, 0.0, 0.02)),
        HexapodTorque.zero(),
    ) * 20

    first_trace = tuple(first.step(item) for item in schedule)
    second_trace = tuple(second.step(item) for item in schedule)

    assert first_trace == second_trace


def test_tripod_phases_remain_half_cycle_apart() -> None:
    simulator = ReferenceHexapod(HexapodParameters(phase_rate_hz=7.0))

    for _ in range(100):
        body = simulator.step(HexapodTorque.zero())

    assert body.tripod_phase_error < 1e-12
    for a, b in (
        ("left_fore", "right_fore"),
        ("right_middle", "left_middle"),
        ("left_hind", "right_hind"),
    ):
        phase_delta = (body.leg(b).phase - body.leg(a).phase) % 1.0
        assert phase_delta == pytest.approx(0.5)


@settings(max_examples=40, deadline=None)
@given(
    torques=st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=18,
        max_size=18,
    )
)
def test_one_step_preserves_finite_bounded_state(torques: list[float]) -> None:
    simulator = ReferenceHexapod()
    grouped = tuple(
        tuple(torques[index : index + 3]) for index in range(0, len(torques), 3)
    )

    body = simulator.step(HexapodTorque(values=grouped))

    assert isinstance(body, HexapodBody)
    assert math.isfinite(body.energy_j)
    assert math.isfinite(body.support_margin_m)
    for leg in body.legs:
        assert all(math.isfinite(value) for value in leg.joint_angles_rad)
        assert all(math.isfinite(value) for value in leg.joint_velocities_rad_s)
        assert all(math.isfinite(value) for value in leg.foot_position_m)
        assert math.isfinite(leg.load_n)
