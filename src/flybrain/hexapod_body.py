"""Deterministic, evidence-bounded reference physics for a six-legged body."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LegName = Literal[
    "left_fore",
    "right_fore",
    "left_middle",
    "right_middle",
    "left_hind",
    "right_hind",
]
JointName = Literal["thorax_coxa", "trochanter", "tibia"]
Vec3 = tuple[float, float, float]

LEG_NAMES: tuple[LegName, ...] = (
    "left_fore",
    "right_fore",
    "left_middle",
    "right_middle",
    "left_hind",
    "right_hind",
)
JOINT_NAMES: tuple[JointName, ...] = ("thorax_coxa", "trochanter", "tibia")
TRIPOD_A: tuple[LegName, ...] = ("left_fore", "right_middle", "left_hind")
TRIPOD_B: tuple[LegName, ...] = ("right_fore", "left_middle", "right_hind")
_MIRROR_LEG: dict[LegName, LegName] = {
    "left_fore": "right_fore",
    "right_fore": "left_fore",
    "left_middle": "right_middle",
    "right_middle": "left_middle",
    "left_hind": "right_hind",
    "right_hind": "left_hind",
}


def _require_finite(values: Iterable[float], label: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")


def _mirror_vector(values: Vec3) -> Vec3:
    return (values[0], -values[1], values[2])


def _mirror_joint_vector(values: Vec3) -> Vec3:
    return (-values[0], values[1], values[2])


class LegState(BaseModel, frozen=True):
    """Immutable joint, contact, and phase state for one named leg."""

    model_config = ConfigDict(extra="forbid")

    name: LegName
    joint_angles_rad: Vec3
    joint_velocities_rad_s: Vec3
    applied_torques_nm: Vec3
    foot_position_m: Vec3
    contact: bool
    load_n: float = Field(ge=0.0)
    phase: float = Field(ge=0.0, lt=1.0)

    @field_validator(
        "joint_angles_rad",
        "joint_velocities_rad_s",
        "applied_torques_nm",
        "foot_position_m",
    )
    @classmethod
    def validate_finite_vector(cls, value: Vec3) -> Vec3:
        _require_finite(value, "leg vector")
        return value

    @field_validator("load_n", "phase")
    @classmethod
    def validate_finite_scalar(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("leg scalar must be finite")
        return value


class HexapodBody(BaseModel, frozen=True):
    """Complete observable reference-body state without task or reward fields."""

    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(ge=0.0)
    thorax_position_m: Vec3
    thorax_velocity_m_s: Vec3
    thorax_yaw_rad: float
    thorax_yaw_rate_rad_s: float
    legs: tuple[LegState, ...]
    energy_j: float = Field(ge=0.0)
    support_count: int = Field(ge=0, le=6)
    support_margin_m: float
    tripod_phase_error: float = Field(ge=0.0)
    fallen: bool

    @field_validator("thorax_position_m", "thorax_velocity_m_s")
    @classmethod
    def validate_finite_vector(cls, value: Vec3) -> Vec3:
        _require_finite(value, "thorax vector")
        return value

    @field_validator(
        "time_s",
        "thorax_yaw_rad",
        "thorax_yaw_rate_rad_s",
        "energy_j",
        "support_margin_m",
        "tripod_phase_error",
    )
    @classmethod
    def validate_finite_scalar(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("body scalar must be finite")
        return value

    @model_validator(mode="after")
    def validate_leg_inventory(self) -> Self:
        names = tuple(leg.name for leg in self.legs)
        if names != LEG_NAMES:
            raise ValueError(f"legs must follow canonical order: {LEG_NAMES}")
        if self.support_count != sum(leg.contact for leg in self.legs):
            raise ValueError("support count disagrees with leg contacts")
        return self

    def leg(self, name: LegName) -> LegState:
        """Return one named leg from the canonical immutable inventory."""

        return self.legs[LEG_NAMES.index(name)]

    def mirror(self) -> HexapodBody:
        """Reflect the entire state across the sagittal plane."""

        mirrored: list[LegState] = []
        for target_name in LEG_NAMES:
            source = self.leg(_MIRROR_LEG[target_name])
            mirrored.append(
                LegState(
                    name=target_name,
                    joint_angles_rad=_mirror_joint_vector(source.joint_angles_rad),
                    joint_velocities_rad_s=_mirror_joint_vector(
                        source.joint_velocities_rad_s
                    ),
                    applied_torques_nm=_mirror_joint_vector(
                        source.applied_torques_nm
                    ),
                    foot_position_m=_mirror_vector(source.foot_position_m),
                    contact=source.contact,
                    load_n=source.load_n,
                    phase=source.phase,
                )
            )
        return HexapodBody(
            time_s=self.time_s,
            thorax_position_m=_mirror_vector(self.thorax_position_m),
            thorax_velocity_m_s=_mirror_vector(self.thorax_velocity_m_s),
            thorax_yaw_rad=-self.thorax_yaw_rad,
            thorax_yaw_rate_rad_s=-self.thorax_yaw_rate_rad_s,
            legs=tuple(mirrored),
            energy_j=self.energy_j,
            support_count=self.support_count,
            support_margin_m=self.support_margin_m,
            tripod_phase_error=self.tripod_phase_error,
            fallen=self.fallen,
        )


class HexapodParameters(BaseModel, frozen=True):
    """Serialized model assumptions for deterministic reference physics."""

    model_config = ConfigDict(extra="forbid")

    dt_s: float = Field(default=0.001, gt=0.0, le=0.01)
    joint_lower_rad: Vec3 = (-0.8, -1.2, -1.6)
    joint_upper_rad: Vec3 = (0.8, 1.2, 0.3)
    initial_joint_angles_rad: Vec3 = (0.0, 0.0, 0.0)
    initial_thorax_position_m: tuple[float, float] = (0.0, 0.0)
    joint_inertia_kg_m2: Vec3 = (2.0e-5, 3.0e-5, 2.0e-5)
    joint_damping_nm_s_rad: Vec3 = (2.0e-4, 3.0e-4, 2.0e-4)
    max_torque_nm: Vec3 = (0.02, 0.03, 0.03)
    link_lengths_m: Vec3 = (0.12, 0.40, 0.40)
    hip_offsets_m: tuple[Vec3, ...] = (
        (0.45, 0.25, 0.0),
        (0.45, -0.25, 0.0),
        (0.0, 0.28, 0.0),
        (0.0, -0.28, 0.0),
        (-0.45, 0.25, 0.0),
        (-0.45, -0.25, 0.0),
    )
    thorax_height_m: float = Field(default=0.79, gt=0.0)
    body_mass_kg: float = Field(default=0.001, gt=0.0)
    yaw_inertia_kg_m2: float = Field(default=0.0002, gt=0.0)
    body_linear_damping_n_s_m: float = Field(default=0.002, ge=0.0)
    body_yaw_damping_nm_s_rad: float = Field(default=0.0002, ge=0.0)
    ground_z_m: float = 0.0
    contact_stiffness_n_m: float = Field(default=0.02, ge=0.0)
    contact_normal_damping_n_s_m: float = Field(default=0.002, ge=0.0)
    contact_tangent_damping_n_s_m: float = Field(default=0.001, ge=0.0)
    friction_coefficient: float = Field(default=0.3, ge=0.0)
    phase_rate_hz: float = Field(default=5.0, ge=0.0)
    fall_support_margin_m: float = -0.02
    joint_mirror_signs: Vec3 = (-1.0, 1.0, 1.0)

    @model_validator(mode="after")
    def validate_parameter_semantics(self) -> Self:
        vectors = (
            self.joint_lower_rad,
            self.joint_upper_rad,
            self.initial_joint_angles_rad,
            self.joint_inertia_kg_m2,
            self.joint_damping_nm_s_rad,
            self.max_torque_nm,
            self.link_lengths_m,
            self.joint_mirror_signs,
        )
        for vector in vectors:
            _require_finite(vector, "parameter vector")
        for offset in self.hip_offsets_m:
            _require_finite(offset, "hip offset")
        if len(self.hip_offsets_m) != len(LEG_NAMES):
            raise ValueError("one hip offset is required per leg")
        if any(
            lower >= initial or initial >= upper
            for lower, initial, upper in zip(
                self.joint_lower_rad,
                self.initial_joint_angles_rad,
                self.joint_upper_rad,
                strict=True,
            )
        ):
            raise ValueError("initial joint angles must lie strictly within limits")
        if any(value <= 0.0 for value in self.joint_inertia_kg_m2):
            raise ValueError("joint inertia must be positive")
        if any(value < 0.0 for value in self.joint_damping_nm_s_rad):
            raise ValueError("joint damping cannot be negative")
        if any(value <= 0.0 for value in self.max_torque_nm):
            raise ValueError("maximum torque must be positive")
        if any(value <= 0.0 for value in self.link_lengths_m):
            raise ValueError("link lengths must be positive")
        if self.joint_mirror_signs != (-1.0, 1.0, 1.0):
            raise ValueError("joint mirror signs must preserve sagittal reflection")
        scalar_values = (
            self.thorax_height_m,
            self.body_mass_kg,
            self.yaw_inertia_kg_m2,
            self.body_linear_damping_n_s_m,
            self.body_yaw_damping_nm_s_rad,
            self.ground_z_m,
            self.contact_stiffness_n_m,
            self.contact_normal_damping_n_s_m,
            self.contact_tangent_damping_n_s_m,
            self.friction_coefficient,
            self.phase_rate_hz,
            self.fall_support_margin_m,
        )
        _require_finite(scalar_values, "parameter scalar")
        _require_finite(self.initial_thorax_position_m, "initial thorax position")
        return self


class HexapodTorque(BaseModel, frozen=True):
    """One ordered three-joint torque vector per canonical leg."""

    model_config = ConfigDict(extra="forbid")

    values: tuple[Vec3, ...]

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: tuple[Vec3, ...]) -> tuple[Vec3, ...]:
        if len(value) != len(LEG_NAMES):
            raise ValueError("one torque vector is required per leg")
        for vector in value:
            _require_finite(vector, "torque vector")
        return value

    @classmethod
    def zero(cls) -> HexapodTorque:
        return cls(values=((0.0, 0.0, 0.0),) * len(LEG_NAMES))

    @classmethod
    def for_leg(cls, name: LegName, values: Vec3) -> HexapodTorque:
        torques = [(0.0, 0.0, 0.0)] * len(LEG_NAMES)
        torques[LEG_NAMES.index(name)] = values
        return cls(values=tuple(torques))

    def mirror(self) -> HexapodTorque:
        mirrored = []
        for target_name in LEG_NAMES:
            source_index = LEG_NAMES.index(_MIRROR_LEG[target_name])
            mirrored.append(_mirror_joint_vector(self.values[source_index]))
        return HexapodTorque(values=tuple(mirrored))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _raw_foot_position(
    thorax_position: Vec3,
    leg_name: LegName,
    joint_angles: Vec3,
    parameters: HexapodParameters,
) -> Vec3:
    index = LEG_NAMES.index(leg_name)
    hip = parameters.hip_offsets_m[index]
    side = 1.0 if leg_name.startswith("left") else -1.0
    coxa, trochanter, tibia = joint_angles
    coxa_length, femur_length, tibia_length = parameters.link_lengths_m
    radial = (
        coxa_length
        + femur_length * math.sin(trochanter)
        + tibia_length * math.sin(trochanter + tibia)
    )
    azimuth = side * math.pi / 2.0 + coxa
    return (
        thorax_position[0] + hip[0] + radial * math.cos(azimuth),
        thorax_position[1] + hip[1] + radial * math.sin(azimuth),
        thorax_position[2]
        + hip[2]
        - femur_length * math.cos(trochanter)
        - tibia_length * math.cos(trochanter + tibia),
    )


def _distance_to_segment(point: tuple[float, float], a: Vec3, b: Vec3) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    projection = _clamp(
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator,
        0.0,
        1.0,
    )
    return math.hypot(
        point[0] - (a[0] + projection * dx),
        point[1] - (a[1] + projection * dy),
    )


def _support_margin(points: list[Vec3], center: tuple[float, float]) -> float:
    unique = sorted({(point[0], point[1], point[2]) for point in points})
    if len(unique) < 3:
        return -1.0

    def cross(origin: Vec3, a: Vec3, b: Vec3) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[Vec3] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[Vec3] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return -1.0
    center3 = (center[0], center[1], 0.0)
    inside = all(
        cross(hull[index], hull[(index + 1) % len(hull)], center3) >= -1e-12
        for index in range(len(hull))
    )
    distance = min(
        _distance_to_segment(center, hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )
    return distance if inside else -distance


def _circular_distance(a: float, b: float) -> float:
    difference = abs(a - b) % 1.0
    return min(difference, 1.0 - difference)


def _tripod_phase_error(legs: tuple[LegState, ...]) -> float:
    phases = {leg.name: leg.phase for leg in legs}
    a_reference = phases[TRIPOD_A[0]]
    b_reference = phases[TRIPOD_B[0]]
    errors = [
        *(_circular_distance(phases[name], a_reference) for name in TRIPOD_A),
        *(_circular_distance(phases[name], b_reference) for name in TRIPOD_B),
        _circular_distance((b_reference - a_reference) % 1.0, 0.5),
    ]
    return max(errors)


class ReferenceHexapod:
    """Small deterministic causal instrument with a replaceable physics boundary."""

    def __init__(self, parameters: HexapodParameters | None = None) -> None:
        self.parameters = parameters or HexapodParameters()
        self._state = self._initial_state()

    def _initial_state(self) -> HexapodBody:
        thorax = (
            self.parameters.initial_thorax_position_m[0],
            self.parameters.initial_thorax_position_m[1],
            self.parameters.thorax_height_m,
        )
        legs = []
        for name in LEG_NAMES:
            raw_foot = _raw_foot_position(
                thorax,
                name,
                self.parameters.initial_joint_angles_rad,
                self.parameters,
            )
            penetration = max(0.0, self.parameters.ground_z_m - raw_foot[2])
            contact = penetration > 0.0
            foot = (raw_foot[0], raw_foot[1], max(raw_foot[2], self.parameters.ground_z_m))
            phase = 0.0 if name in TRIPOD_A else 0.5
            legs.append(
                LegState(
                    name=name,
                    joint_angles_rad=self.parameters.initial_joint_angles_rad,
                    joint_velocities_rad_s=(0.0, 0.0, 0.0),
                    applied_torques_nm=(0.0, 0.0, 0.0),
                    foot_position_m=foot,
                    contact=contact,
                    load_n=self.parameters.contact_stiffness_n_m * penetration,
                    phase=phase,
                )
            )
        leg_tuple = tuple(legs)
        support_points = [leg.foot_position_m for leg in leg_tuple if leg.contact]
        margin = _support_margin(support_points, (0.0, 0.0))
        support_count = len(support_points)
        return HexapodBody(
            time_s=0.0,
            thorax_position_m=thorax,
            thorax_velocity_m_s=(0.0, 0.0, 0.0),
            thorax_yaw_rad=0.0,
            thorax_yaw_rate_rad_s=0.0,
            legs=leg_tuple,
            energy_j=0.0,
            support_count=support_count,
            support_margin_m=margin,
            tripod_phase_error=_tripod_phase_error(leg_tuple),
            fallen=support_count < 3 or margin < self.parameters.fall_support_margin_m,
        )

    def observe(self) -> HexapodBody:
        """Return the current immutable state."""

        return self._state

    def step(self, requested_torque: HexapodTorque) -> HexapodBody:
        """Advance one deterministic semi-implicit integration step."""

        dt = self.parameters.dt_s
        previous = self._state
        provisional: list[tuple[LegName, Vec3, Vec3, Vec3, float, Vec3, float, bool]] = []
        net_force_x = -self.parameters.body_linear_damping_n_s_m * previous.thorax_velocity_m_s[0]
        net_force_y = -self.parameters.body_linear_damping_n_s_m * previous.thorax_velocity_m_s[1]
        net_yaw_torque = (
            -self.parameters.body_yaw_damping_nm_s_rad * previous.thorax_yaw_rate_rad_s
        )
        energy_delta = 0.0

        for index, name in enumerate(LEG_NAMES):
            old_leg = previous.legs[index]
            requested = requested_torque.values[index]
            limits = self.parameters.max_torque_nm
            applied: Vec3 = (
                _clamp(requested[0], -limits[0], limits[0]),
                _clamp(requested[1], -limits[1], limits[1]),
                _clamp(requested[2], -limits[2], limits[2]),
            )
            velocity_values = []
            angle_values = []
            for joint in range(3):
                acceleration = (
                    applied[joint]
                    - self.parameters.joint_damping_nm_s_rad[joint]
                    * old_leg.joint_velocities_rad_s[joint]
                ) / self.parameters.joint_inertia_kg_m2[joint]
                velocity = old_leg.joint_velocities_rad_s[joint] + acceleration * dt
                angle = old_leg.joint_angles_rad[joint] + velocity * dt
                lower = self.parameters.joint_lower_rad[joint]
                upper = self.parameters.joint_upper_rad[joint]
                if angle < lower:
                    angle = lower
                    velocity = 0.0
                elif angle > upper:
                    angle = upper
                    velocity = 0.0
                velocity_values.append(velocity)
                angle_values.append(angle)
                energy_delta += abs(applied[joint] * velocity) * dt
            angles: Vec3 = (angle_values[0], angle_values[1], angle_values[2])
            velocities: Vec3 = (
                velocity_values[0],
                velocity_values[1],
                velocity_values[2],
            )
            old_raw = _raw_foot_position(
                previous.thorax_position_m,
                name,
                old_leg.joint_angles_rad,
                self.parameters,
            )
            new_raw = _raw_foot_position(
                previous.thorax_position_m,
                name,
                angles,
                self.parameters,
            )
            foot_velocity = tuple(
                (new - old) / dt for new, old in zip(new_raw, old_raw, strict=True)
            )
            penetration = max(0.0, self.parameters.ground_z_m - new_raw[2])
            load = max(
                0.0,
                self.parameters.contact_stiffness_n_m * penetration
                + self.parameters.contact_normal_damping_n_s_m
                * max(0.0, -foot_velocity[2]),
            )
            contact = penetration > 0.0
            if contact:
                friction_limit = self.parameters.friction_coefficient * load
                force_x = _clamp(
                    -self.parameters.contact_tangent_damping_n_s_m * foot_velocity[0],
                    -friction_limit,
                    friction_limit,
                )
                force_y = _clamp(
                    -self.parameters.contact_tangent_damping_n_s_m * foot_velocity[1],
                    -friction_limit,
                    friction_limit,
                )
                net_force_x += force_x
                net_force_y += force_y
                lever_x = new_raw[0] - previous.thorax_position_m[0]
                lever_y = new_raw[1] - previous.thorax_position_m[1]
                net_yaw_torque += lever_x * force_y - lever_y * force_x
            phase = (old_leg.phase + self.parameters.phase_rate_hz * dt) % 1.0
            provisional.append(
                (name, angles, velocities, applied, phase, new_raw, load, contact)
            )

        velocity_x = previous.thorax_velocity_m_s[0] + (
            net_force_x / self.parameters.body_mass_kg
        ) * dt
        velocity_y = previous.thorax_velocity_m_s[1] + (
            net_force_y / self.parameters.body_mass_kg
        ) * dt
        yaw_rate = previous.thorax_yaw_rate_rad_s + (
            net_yaw_torque / self.parameters.yaw_inertia_kg_m2
        ) * dt
        thorax_position = (
            previous.thorax_position_m[0] + velocity_x * dt,
            previous.thorax_position_m[1] + velocity_y * dt,
            self.parameters.thorax_height_m,
        )
        yaw = previous.thorax_yaw_rad + yaw_rate * dt

        legs = []
        for name, angles, velocities, applied, phase, _, load, contact in provisional:
            raw_foot = _raw_foot_position(thorax_position, name, angles, self.parameters)
            foot = (
                raw_foot[0],
                raw_foot[1],
                max(raw_foot[2], self.parameters.ground_z_m),
            )
            legs.append(
                LegState(
                    name=name,
                    joint_angles_rad=angles,
                    joint_velocities_rad_s=velocities,
                    applied_torques_nm=applied,
                    foot_position_m=foot,
                    contact=contact,
                    load_n=load,
                    phase=phase,
                )
            )
        leg_tuple = tuple(legs)
        support_points = [leg.foot_position_m for leg in leg_tuple if leg.contact]
        support_count = len(support_points)
        support_margin = _support_margin(
            support_points, (thorax_position[0], thorax_position[1])
        )
        self._state = HexapodBody(
            time_s=previous.time_s + dt,
            thorax_position_m=thorax_position,
            thorax_velocity_m_s=(velocity_x, velocity_y, 0.0),
            thorax_yaw_rad=yaw,
            thorax_yaw_rate_rad_s=yaw_rate,
            legs=leg_tuple,
            energy_j=previous.energy_j + energy_delta,
            support_count=support_count,
            support_margin_m=support_margin,
            tripod_phase_error=_tripod_phase_error(leg_tuple),
            fallen=(
                support_count < 3
                or support_margin < self.parameters.fall_support_margin_m
            ),
        )
        return self._state
