import math

import pytest

from flybrain.flygym_backend import FlyGymBackend, flygym_availability
from flybrain.hexapod_backend import (
    ReferenceHexapodBackend,
    run_backend_parity,
    run_backend_trace,
)
from flybrain.hexapod_body import LEG_NAMES, HexapodParameters, HexapodTorque

availability = flygym_availability()
pytestmark = pytest.mark.skipif(not availability.available, reason=availability.reason)


def test_flygym_backend_implements_shared_torque_observation_contract() -> None:
    parameters = HexapodParameters(dt_s=0.001)
    backend = FlyGymBackend(parameters)

    initial = backend.observe()
    stepped = backend.step(HexapodTorque.zero())

    assert backend.identity.name == "flygym_mujoco"
    assert backend.identity.version == "flygym-2.1.0"
    assert backend.identity.dt_s == 0.001
    assert tuple(leg.name for leg in initial.legs) == LEG_NAMES
    assert stepped.time_s == pytest.approx(0.001)
    assert all(math.isfinite(value) for leg in stepped.legs for value in leg.joint_angles_rad)


def test_flygym_backend_uses_flygym_physics_scale_and_substeps_body_windows() -> None:
    backend = FlyGymBackend(HexapodParameters(dt_s=0.01))

    assert backend._physics_substeps == 100
    assert backend._physics_dt_s == pytest.approx(0.0001)
    assert (backend._simulation.mj_model.actuator_forcerange[:, 1] <= 65.0).all()


def test_flygym_backend_preserves_six_distinct_foot_positions() -> None:
    body = FlyGymBackend(HexapodParameters(dt_s=0.001)).observe()

    assert len({leg.foot_position_m for leg in body.legs}) == 6


def test_flygym_backend_honors_world_initial_position_in_meters() -> None:
    origin = FlyGymBackend(HexapodParameters(dt_s=0.001)).observe()
    shifted = FlyGymBackend(
        HexapodParameters(
            dt_s=0.001,
            initial_thorax_position_m=(0.003, -0.002),
        )
    ).observe()

    assert shifted.thorax_position_m[0] - origin.thorax_position_m[0] == pytest.approx(0.003)
    assert shifted.thorax_position_m[1] - origin.thorax_position_m[1] == pytest.approx(-0.002)


def test_flygym_backend_reports_measured_clipped_torque_and_work() -> None:
    backend = FlyGymBackend(HexapodParameters(dt_s=0.001))
    body = backend.step(HexapodTorque.for_leg("left_fore", (0.001, 0.0, 0.0)))

    assert body.leg("left_fore").applied_torques_nm[0] == pytest.approx(65e-6)
    requested_work_estimate = 0.001 * abs(
        body.leg("left_fore").joint_velocities_rad_s[0]
    ) * 0.001
    assert 0.0 < body.energy_j < requested_work_estimate / 10.0


def test_flygym_backend_applies_mass_and_friction_holdout_perturbations() -> None:
    baseline = FlyGymBackend(HexapodParameters(dt_s=0.001))
    changed = FlyGymBackend(
        HexapodParameters(
            dt_s=0.001,
            body_mass_kg=0.00105,
            friction_coefficient=0.27,
        )
    )

    baseline_mass = baseline._simulation.mj_model.body_mass.sum()
    changed_mass = changed._simulation.mj_model.body_mass.sum()
    assert changed_mass / baseline_mass == pytest.approx(1.05)
    baseline_friction = baseline._simulation.mj_model.geom_friction[0, 0]
    changed_friction = changed._simulation.mj_model.geom_friction[0, 0]
    assert changed_friction / baseline_friction == pytest.approx(0.9)


def test_flygym_backend_reset_and_replay_are_exact_on_same_platform() -> None:
    parameters = HexapodParameters(dt_s=0.001)
    schedule = (
        HexapodTorque.for_leg("left_fore", (1e-6, 2e-6, -2e-6)),
        HexapodTorque.zero(),
    )
    backend = FlyGymBackend(parameters)
    initial = backend.observe()

    first = run_backend_trace(backend, schedule)
    restored = backend.reset()
    second = run_backend_trace(backend, schedule)

    assert restored == initial
    assert first.observations == second.observations
    assert first.trace_digest == second.trace_digest


def test_flygym_backend_maps_each_canonical_leg_to_three_motor_dofs() -> None:
    backend = FlyGymBackend(HexapodParameters(dt_s=0.001))

    assert backend.actuated_dof_names == (
        "c_thorax-lf_coxa-yaw",
        "lf_coxa-lf_trochanterfemur-pitch",
        "lf_trochanterfemur-lf_tibia-pitch",
        "c_thorax-rf_coxa-yaw",
        "rf_coxa-rf_trochanterfemur-pitch",
        "rf_trochanterfemur-rf_tibia-pitch",
        "c_thorax-lm_coxa-yaw",
        "lm_coxa-lm_trochanterfemur-pitch",
        "lm_trochanterfemur-lm_tibia-pitch",
        "c_thorax-rm_coxa-yaw",
        "rm_coxa-rm_trochanterfemur-pitch",
        "rm_trochanterfemur-rm_tibia-pitch",
        "c_thorax-lh_coxa-yaw",
        "lh_coxa-lh_trochanterfemur-pitch",
        "lh_trochanterfemur-lh_tibia-pitch",
        "c_thorax-rh_coxa-yaw",
        "rh_coxa-rh_trochanterfemur-pitch",
        "rh_trochanterfemur-rh_tibia-pitch",
    )


def test_flygym_availability_records_dependency_and_platform() -> None:
    assert availability.available is True
    assert availability.package_version == "2.1.0"
    assert availability.platform
    assert availability.python_version


def test_reference_and_flygym_preserve_all_joint_response_directions() -> None:
    parameters = HexapodParameters(dt_s=0.001)
    pulse = HexapodTorque(values=((1e-6, 1e-6, 1e-6),) * 6)

    result = run_backend_parity(
        ReferenceHexapodBackend(parameters),
        FlyGymBackend(parameters),
        (pulse, pulse, pulse),
    )

    assert result.classification == "positive"
    assert result.compared_joints == 18
    assert result.direction_matches == 18
    assert result.reference_replay_exact is True
    assert result.candidate_replay_exact is True
    assert result.identical_torque_schedule is True
