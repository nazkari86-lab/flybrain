"""Optional FlyGym/MuJoCo implementation of the shared hexapod backend."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math
import platform
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from flybrain.hexapod_backend import BackendIdentity, _validate_torque
from flybrain.hexapod_body import (
    LEG_NAMES,
    TRIPOD_A,
    HexapodBody,
    HexapodParameters,
    HexapodTorque,
    LegState,
    Vec3,
    _support_margin,
)

_LEG_CODE = {
    "left_fore": "lf",
    "right_fore": "rf",
    "left_middle": "lm",
    "right_middle": "rm",
    "left_hind": "lh",
    "right_hind": "rh",
}
_FLYGYM_LEG_INDEX = {name: index for index, name in enumerate(("lf", "lm", "lh", "rf", "rm", "rh"))}
_NM_TO_FLYGYM_TORQUE = 1e6
_FLYGYM_FORCE_TO_N = 1e-3
_MM_TO_M = 1e-3
_FLYGYM_MAX_PHYSICS_DT_S = 1e-4
_FLYGYM_ACTUATOR_FORCE_LIMIT = 65.0


class FlyGymAvailability(BaseModel, frozen=True):
    """Auditable optional-dependency and host-platform status."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    package_version: str | None
    platform: str
    python_version: str
    reason: str


def flygym_availability() -> FlyGymAvailability:
    """Report whether the pinned FlyGym backend can be imported on this host."""

    present = importlib.util.find_spec("flygym") is not None
    version = importlib.metadata.version("flygym") if present else None
    compatible = present and version == "2.1.0"
    reason = "available" if compatible else (
        "FlyGym is not installed; install the physics extra"
        if not present
        else f"FlyGym 2.1.0 required, found {version}"
    )
    return FlyGymAvailability(
        available=compatible,
        package_version=version,
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        reason=reason,
    )


def _dof_names() -> tuple[str, ...]:
    names: list[str] = []
    for leg in LEG_NAMES:
        code = _LEG_CODE[leg]
        names.extend(
            (
                f"c_thorax-{code}_coxa-yaw",
                f"{code}_coxa-{code}_trochanterfemur-pitch",
                f"{code}_trochanterfemur-{code}_tibia-pitch",
            )
        )
    return tuple(names)


def _yaw_from_quaternion(quaternion: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _vec3(values: Any, *, scale: float = 1.0) -> Vec3:
    return (
        float(values[0]) * scale,
        float(values[1]) * scale,
        float(values[2]) * scale,
    )


class FlyGymBackend:
    """FlyGym 2.1 adapter with no policy, target, or reward interface."""

    def __init__(self, parameters: HexapodParameters | None = None) -> None:
        status = flygym_availability()
        if not status.available:
            raise RuntimeError(status.reason)
        self._parameters = parameters or HexapodParameters()
        self._dof_names = _dof_names()
        self._flygym = importlib.import_module("flygym")
        self._compose = importlib.import_module("flygym.compose")
        self._anatomy = importlib.import_module("flygym.anatomy")
        self._math = importlib.import_module("flygym.utils.math")
        self._mujoco = importlib.import_module("mujoco")
        self._physics_substeps = max(
            1, math.ceil(self._parameters.dt_s / _FLYGYM_MAX_PHYSICS_DT_S)
        )
        self._physics_dt_s = self._parameters.dt_s / self._physics_substeps
        self._build()

    def _build(self) -> None:
        fly = self._compose.NeuroMechFly(name="flybrain")
        axis_order = self._anatomy.AxisOrder.YAW_PITCH_ROLL
        skeleton = self._anatomy.Skeleton(
            axis_order=axis_order,
            joint_preset=self._anatomy.JointPreset.LEGS_ONLY,
        )
        neutral_pose = self._compose.KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            axis_order
        )
        fly.add_joints(skeleton, neutral_pose)
        by_name = {item.name: item for item in fly.get_jointdofs_order()}
        missing = sorted(set(self._dof_names) - set(by_name))
        if missing:
            raise RuntimeError(f"FlyGym joint map is incomplete: {missing}")
        selected = [by_name[name] for name in self._dof_names]
        largest_authority = min(
            max(self._parameters.max_torque_nm) * _NM_TO_FLYGYM_TORQUE,
            _FLYGYM_ACTUATOR_FORCE_LIMIT,
        )
        fly.add_actuators(
            selected,
            self._compose.ActuatorType.MOTOR,
            forcerange=(-largest_authority, largest_authority),
        )
        world = self._compose.FlatGroundWorld()
        world.add_fly(
            fly,
            (
                self._parameters.initial_thorax_position_m[0] / _MM_TO_M,
                self._parameters.initial_thorax_position_m[1] / _MM_TO_M,
                0.5,
            ),
            self._math.Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        self._fly = fly
        self._simulation = self._flygym.Simulation(
            world,
            timestep=self._physics_dt_s,
        )
        default_parameters = HexapodParameters()
        mass_ratio = self._parameters.body_mass_kg / default_parameters.body_mass_kg
        friction_ratio = (
            self._parameters.friction_coefficient
            / default_parameters.friction_coefficient
        )
        model = self._simulation.mj_model
        model.body_mass[:] *= mass_ratio
        model.body_inertia[:] *= mass_ratio
        model.geom_friction[:, 0] *= friction_ratio
        self._mujoco.mj_setConst(model, self._simulation.mj_data)
        self._all_dof_names = tuple(item.name for item in fly.get_jointdofs_order())
        self._joint_indices = tuple(self._all_dof_names.index(name) for name in self._dof_names)
        body_names = tuple(item.name for item in fly.get_bodysegs_order())
        self._thorax_index = body_names.index("c_thorax")
        self._foot_indices = {
            leg: body_names.index(f"{_LEG_CODE[leg]}_tarsus5") for leg in LEG_NAMES
        }
        self._time_s = 0.0
        self._energy_j = 0.0
        self._last_torque = HexapodTorque.zero()
        self._mujoco.mj_forward(self._simulation.mj_model, self._simulation.mj_data)
        self._previous_position_m = self._thorax_position_m()
        self._previous_yaw_rad = self._thorax_yaw_rad()

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(
            name="flygym_mujoco",
            version="flygym-2.1.0",
            dt_s=self._parameters.dt_s,
        )

    @property
    def parameters(self) -> HexapodParameters:
        return self._parameters

    @property
    def actuated_dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    def _thorax_position_m(self) -> tuple[float, float, float]:
        values = self._simulation.get_body_positions("flybrain")[self._thorax_index]
        return _vec3(values, scale=_MM_TO_M)

    def _thorax_yaw_rad(self) -> float:
        quaternion = self._simulation.get_body_rotations("flybrain")[self._thorax_index]
        return _yaw_from_quaternion(quaternion)

    def observe(self) -> HexapodBody:
        joint_angles = self._simulation.get_joint_angles("flybrain")
        joint_velocities = self._simulation.get_joint_velocities("flybrain")
        body_positions = self._simulation.get_body_positions("flybrain")
        contact, forces, _, _, _, _ = self._simulation.get_ground_contact_info("flybrain")
        position = self._thorax_position_m()
        yaw = self._thorax_yaw_rad()
        dt = self._parameters.dt_s
        velocity = _vec3(
            [
                (current - previous) / dt
                for current, previous in zip(
                    position, self._previous_position_m, strict=True
                )
            ]
        )
        yaw_rate = (yaw - self._previous_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
        yaw_rate /= dt
        legs = []
        support_points: list[Vec3] = []
        for leg_index, leg in enumerate(LEG_NAMES):
            start = leg_index * 3
            indices = self._joint_indices[start : start + 3]
            angles = _vec3([joint_angles[index] for index in indices])
            velocities = _vec3([joint_velocities[index] for index in indices])
            applied = self._last_torque.values[leg_index]
            foot_values = body_positions[self._foot_indices[leg]]
            foot = _vec3(foot_values, scale=_MM_TO_M)
            flygym_index = _FLYGYM_LEG_INDEX[_LEG_CODE[leg]]
            touching = bool(contact[flygym_index] > 0.5)
            load = float(np.linalg.norm(forces[flygym_index])) * _FLYGYM_FORCE_TO_N
            phase_offset = 0.0 if leg in TRIPOD_A else 0.5
            phase = (phase_offset + self._time_s * self._parameters.phase_rate_hz) % 1.0
            if touching:
                support_points.append(foot)
            legs.append(
                LegState(
                    name=leg,
                    joint_angles_rad=angles,
                    joint_velocities_rad_s=velocities,
                    applied_torques_nm=applied,
                    foot_position_m=foot,
                    contact=touching,
                    load_n=max(0.0, load),
                    phase=phase,
                )
            )
        margin = _support_margin(support_points, (position[0], position[1]))
        support_count = len(support_points)
        return HexapodBody(
            time_s=self._time_s,
            thorax_position_m=position,
            thorax_velocity_m_s=velocity,
            thorax_yaw_rad=yaw,
            thorax_yaw_rate_rad_s=yaw_rate,
            legs=tuple(legs),
            energy_j=self._energy_j,
            support_count=support_count,
            support_margin_m=margin,
            tripod_phase_error=0.0,
            fallen=support_count < 3 or margin < self._parameters.fall_support_margin_m,
        )

    def step(self, torque: HexapodTorque) -> HexapodBody:
        _validate_torque(torque, self._parameters)
        flattened = np.asarray(
            [value * _NM_TO_FLYGYM_TORQUE for vector in torque.values for value in vector],
            dtype=np.float64,
        )
        self._simulation.set_actuator_inputs(
            "flybrain",
            self._compose.ActuatorType.MOTOR,
            flattened,
        )
        before = self.observe()
        for _ in range(self._physics_substeps):
            self._simulation.step()
            forces_nm = (
                self._simulation.get_actuator_forces(
                    "flybrain", self._compose.ActuatorType.MOTOR
                )
                / _NM_TO_FLYGYM_TORQUE
            )
            velocities = self._simulation.get_joint_velocities("flybrain")
            self._energy_j += sum(
                abs(float(force) * float(velocities[index])) * self._physics_dt_s
                for force, index in zip(forces_nm, self._joint_indices, strict=True)
            )
        self._time_s += self._parameters.dt_s
        self._last_torque = HexapodTorque(
            values=tuple(
                _vec3(forces_nm[start : start + 3])
                for start in range(0, len(forces_nm), 3)
            )
        )
        self._previous_position_m = before.thorax_position_m
        self._previous_yaw_rad = before.thorax_yaw_rad
        return self.observe()

    def reset(self) -> HexapodBody:
        self._simulation.reset()
        self._mujoco.mj_forward(self._simulation.mj_model, self._simulation.mj_data)
        self._time_s = 0.0
        self._energy_j = 0.0
        self._last_torque = HexapodTorque.zero()
        self._previous_position_m = self._thorax_position_m()
        self._previous_yaw_rad = self._thorax_yaw_rad()
        return self.observe()
