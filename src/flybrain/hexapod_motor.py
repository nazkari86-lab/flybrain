"""Registry-backed motor population mapping and bounded spike decoding."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flybrain.biological_registry import ResolvedRegistry
from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import LEG_NAMES, HexapodTorque, LegName, Vec3

MotorJoint = Literal["thorax_coxa", "trochanter", "tibia"]
MotorDirection = Literal["anterior", "posterior", "flexor", "extensor"]


def _canonical_descriptors() -> tuple[tuple[str, LegName, MotorJoint, MotorDirection], ...]:
    joint_directions: tuple[tuple[MotorJoint, tuple[MotorDirection, MotorDirection]], ...] = (
        ("thorax_coxa", ("anterior", "posterior")),
        ("trochanter", ("flexor", "extensor")),
        ("tibia", ("flexor", "extensor")),
    )
    return tuple(
        (f"{leg}_{joint}_{direction}", leg, joint, direction)
        for leg in LEG_NAMES
        for joint, directions in joint_directions
        for direction in directions
    )


CANONICAL_MOTOR_GROUPS = _canonical_descriptors()


class MotorGroup(BaseModel, frozen=True):
    """One exact registry population with a single antagonistic channel role."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    leg: LegName
    joint: MotorJoint
    direction: MotorDirection
    neuron_ids: tuple[int, ...]

    @field_validator("neuron_ids", mode="before")
    @classmethod
    def validate_id_types(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not int for item in value
        ):
            raise ValueError("motor population must contain integer IDs")
        return value

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        expected_name = f"{self.leg}_{self.joint}_{self.direction}"
        if self.name != expected_name:
            raise ValueError(f"motor group name must be {expected_name}")
        if not self.neuron_ids or any(value <= 0 for value in self.neuron_ids):
            raise ValueError("motor population must contain positive IDs")
        if tuple(sorted(set(self.neuron_ids))) != self.neuron_ids:
            raise ValueError("motor population IDs must be unique and sorted")
        return self


class PhaseEnvelopeAssumption(BaseModel, frozen=True):
    """Optional model-only thorax-coxa calibration oscillator."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["model_assumption"] = "model_assumption"
    amplitude_nm: float = Field(default=0.0, ge=0.0)
    waveform: Literal["sine"] = "sine"
    tripod_phase_offsets: tuple[float, float, float, float, float, float] = (
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.5,
    )

    @model_validator(mode="after")
    def validate_phase_assumption(self) -> Self:
        if not math.isfinite(self.amplitude_nm):
            raise ValueError("phase-envelope amplitude must be finite")
        if any(
            not math.isfinite(value) or not 0.0 <= value < 1.0
            for value in self.tripod_phase_offsets
        ):
            raise ValueError("tripod phase offsets must be finite values in [0, 1)")
        expected = (0.0, 0.5, 0.5, 0.0, 0.0, 0.5)
        if self.tripod_phase_offsets != expected:
            raise ValueError("tripod phase offsets must declare the canonical alternating tripods")
        return self


class HexapodMotorMap(BaseModel, frozen=True):
    """Complete, disjoint 36-channel motor map in canonical order."""

    model_config = ConfigDict(extra="forbid")

    groups: tuple[MotorGroup, ...]

    @model_validator(mode="after")
    def validate_complete_map(self) -> Self:
        expected_names = tuple(item[0] for item in CANONICAL_MOTOR_GROUPS)
        names = tuple(group.name for group in self.groups)
        if names != expected_names:
            raise ValueError("motor map must contain the complete canonical group order")
        owner: dict[int, str] = {}
        for group in self.groups:
            overlap = sorted(set(group.neuron_ids) & owner.keys())
            if overlap:
                raise ValueError(
                    f"motor population overlap for {group.name}: {overlap}"
                )
            owner.update(dict.fromkeys(group.neuron_ids, group.name))
        return self

    @classmethod
    def from_registry(cls, registry: ResolvedRegistry) -> HexapodMotorMap:
        groups = []
        for name, leg, joint, direction in CANONICAL_MOTOR_GROUPS:
            population = registry.population(name)
            if population.role != "motor":
                raise ValueError(f"resolved population is not motor: {name}")
            groups.append(
                MotorGroup(
                    name=name,
                    leg=leg,
                    joint=joint,
                    direction=direction,
                    neuron_ids=population.neuron_ids,
                )
            )
        return cls(groups=tuple(groups))

    def group(self, name: str) -> MotorGroup:
        matches = tuple(group for group in self.groups if group.name == name)
        if len(matches) != 1:
            raise ValueError(f"motor group not found exactly once: {name}")
        return matches[0]

    def named_populations(self) -> dict[str, tuple[int, ...]]:
        return {group.name: group.neuron_ids for group in self.groups}

    def validate_graph(self, graph: EventConnectome) -> None:
        available = {int(value) for value in graph.neuron_ids}
        required = {
            neuron_id for group in self.groups for neuron_id in group.neuron_ids
        }
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"motor population IDs absent from graph: {missing}")


class MotorActivationState(BaseModel, frozen=True):
    """Normalized rates, filtered activations, and joint torques for one window."""

    model_config = ConfigDict(extra="forbid")

    group_names: tuple[str, ...]
    rates_hz: tuple[float, ...]
    activations: tuple[float, ...]
    torques: HexapodTorque
    window_s: float = Field(gt=0.0)
    leg_phases: tuple[float, float, float, float, float, float]
    activation_time_constant_s: float = Field(gt=0.0)
    saturation_rate_hz: float = Field(gt=0.0)
    neural_authority_nm: tuple[float, float, float]
    phase_envelope: PhaseEnvelopeAssumption
    descending_phase_gain: float = Field(gt=0.0)
    descending_phase_bias: float

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        expected_names = tuple(item[0] for item in CANONICAL_MOTOR_GROUPS)
        if self.group_names != expected_names:
            raise ValueError("activation state group order is not canonical")
        if len(self.rates_hz) != len(expected_names):
            raise ValueError("activation state rate count differs from motor groups")
        if len(self.activations) != len(expected_names):
            raise ValueError("activation state activation count differs from motor groups")
        scalars = (
            *self.rates_hz,
            *self.activations,
            *self.leg_phases,
            self.window_s,
            self.activation_time_constant_s,
            self.saturation_rate_hz,
            *self.neural_authority_nm,
            self.descending_phase_gain,
            self.descending_phase_bias,
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("motor activation state must be finite")
        if any(value < 0.0 for value in self.rates_hz):
            raise ValueError("motor rates cannot be negative")
        if any(not 0.0 <= value <= 1.0 for value in self.activations):
            raise ValueError("motor activations must lie in [0, 1]")
        if any(not 0.0 <= value < 1.0 for value in self.leg_phases):
            raise ValueError("leg phases must lie in [0, 1)")
        if any(value <= 0.0 for value in self.neural_authority_nm):
            raise ValueError("neural torque authority must be positive")
        if not 0.5 <= self.descending_phase_gain <= 1.5:
            raise ValueError("descending phase gain must lie in [0.5, 1.5]")
        if not -0.25 <= self.descending_phase_bias <= 0.25:
            raise ValueError("descending phase bias must lie in [-0.25, 0.25]")
        return self

    def rate_hz(self, name: str) -> float:
        try:
            return self.rates_hz[self.group_names.index(name)]
        except ValueError as error:
            raise ValueError(f"motor group absent from state: {name}") from error

    def activation(self, name: str) -> float:
        try:
            return self.activations[self.group_names.index(name)]
        except ValueError as error:
            raise ValueError(f"motor group absent from state: {name}") from error


class HexapodMotorDecoder:
    """Decode motor spikes with bounded descending modulation of neural drive."""

    version = "hexapod-motor-decoder-v3"

    def __init__(
        self,
        mapping: HexapodMotorMap,
        *,
        activation_time_constant_s: float = 0.02,
        saturation_rate_hz: float = 100.0,
        neural_authority_nm: tuple[float, float, float] = (0.01, 0.01, 0.01),
        phase_envelope: PhaseEnvelopeAssumption | None = None,
        descending_map: DescendingMap | None = None,
    ) -> None:
        parameters = (
            activation_time_constant_s,
            saturation_rate_hz,
            *neural_authority_nm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in parameters):
            raise ValueError("motor decoder parameters must be finite and positive")
        self.mapping = mapping
        self.activation_time_constant_s = float(activation_time_constant_s)
        self.saturation_rate_hz = float(saturation_rate_hz)
        self.neural_authority_nm = (
            float(neural_authority_nm[0]),
            float(neural_authority_nm[1]),
            float(neural_authority_nm[2]),
        )
        self.phase_envelope = phase_envelope or PhaseEnvelopeAssumption()
        self.descending_map = descending_map
        self._activations = (0.0,) * len(mapping.groups)

    def _descending_phase_modulation(
        self, spikes: tuple[int, ...], *, window_s: float
    ) -> tuple[float, float]:
        if self.descending_map is None:
            return 1.0, 0.0
        counts = Counter(spikes)

        def rate(ids: tuple[int, ...]) -> float:
            hz = sum(counts[value] for value in ids) / (len(ids) * window_s)
            return min(hz / self.saturation_rate_hz, 1.0)

        na_left = rate(self.descending_map.d_na02_left)
        na_right = rate(self.descending_map.d_na02_right)
        ng_left = rate(self.descending_map.d_ng13_left)
        ng_right = rate(self.descending_map.d_ng13_right)
        mdn_left = rate(self.descending_map.mdn_left)
        mdn_right = rate(self.descending_map.mdn_right)
        lateral_drive = 0.25 * (
            (na_left - na_right) + (ng_left - ng_right)
        ) + 0.125 * (mdn_left - mdn_right)
        phase_bias = float(np.clip(lateral_drive, -0.25, 0.25))
        retreat_drive = 0.25 * (mdn_left + mdn_right)
        phase_gain = float(np.clip(1.0 + retreat_drive, 0.5, 1.5))
        return phase_gain, phase_bias

    def decode(
        self,
        spikes: tuple[int, ...],
        *,
        window_s: float,
        leg_phases: tuple[float, float, float, float, float, float],
    ) -> MotorActivationState:
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise ValueError("motor decoding window must be finite and positive")
        if len(leg_phases) != len(LEG_NAMES) or any(
            not math.isfinite(value) or not 0.0 <= value < 1.0
            for value in leg_phases
        ):
            raise ValueError("one finite phase in [0, 1) is required per leg")
        if any(type(value) is not int or value <= 0 for value in spikes):
            raise ValueError("spikes must contain positive integer neuron IDs")

        counts = Counter(spikes)
        descending_phase_gain, descending_phase_bias = (
            self._descending_phase_modulation(spikes, window_s=window_s)
        )
        rates = tuple(
            sum(counts[neuron_id] for neuron_id in group.neuron_ids)
            / (len(group.neuron_ids) * window_s)
            for group in self.mapping.groups
        )
        alpha = 1.0 - math.exp(-window_s / self.activation_time_constant_s)
        activations = tuple(
            previous
            + alpha * (min(rate / self.saturation_rate_hz, 1.0) - previous)
            for previous, rate in zip(self._activations, rates, strict=True)
        )
        self._activations = activations
        activation_by_name = {
            group.name: activation
            for group, activation in zip(self.mapping.groups, activations, strict=True)
        }
        torque_values: list[Vec3] = []
        for leg_index, leg in enumerate(LEG_NAMES):
            side_sign = 1.0 if leg.startswith("left") else -1.0
            lateral_neural_gain = 1.0 + side_sign * descending_phase_bias
            phase = (
                leg_phases[leg_index] + side_sign * descending_phase_bias
            ) % 1.0
            phase_torque = self.phase_envelope.amplitude_nm * math.sin(
                2.0 * math.pi * phase
            ) * descending_phase_gain
            thorax_coxa = phase_torque + (
                descending_phase_gain
                * lateral_neural_gain
                * side_sign
                * self.neural_authority_nm[0]
                * (
                    activation_by_name[f"{leg}_thorax_coxa_anterior"]
                    - activation_by_name[f"{leg}_thorax_coxa_posterior"]
                )
            )
            trochanter = descending_phase_gain * self.neural_authority_nm[1] * (
                activation_by_name[f"{leg}_trochanter_extensor"]
                - activation_by_name[f"{leg}_trochanter_flexor"]
            )
            tibia = descending_phase_gain * self.neural_authority_nm[2] * (
                activation_by_name[f"{leg}_tibia_extensor"]
                - activation_by_name[f"{leg}_tibia_flexor"]
            )
            torque_values.append((thorax_coxa, trochanter, tibia))
        state = MotorActivationState(
            group_names=tuple(group.name for group in self.mapping.groups),
            rates_hz=rates,
            activations=activations,
            torques=HexapodTorque(values=tuple(torque_values)),
            window_s=window_s,
            leg_phases=leg_phases,
            activation_time_constant_s=self.activation_time_constant_s,
            saturation_rate_hz=self.saturation_rate_hz,
            neural_authority_nm=self.neural_authority_nm,
            phase_envelope=self.phase_envelope,
            descending_phase_gain=descending_phase_gain,
            descending_phase_bias=descending_phase_bias,
        )
        return state


def motor_population_silence_mask(
    graph: EventConnectome,
    mapping: HexapodMotorMap,
    names: frozenset[str],
) -> NDArray[np.bool_]:
    """Return an index mask for exactly named, registry-backed motor populations."""

    mapping.validate_graph(graph)
    populations = mapping.named_populations()
    unknown = sorted(names - populations.keys())
    if unknown:
        raise ValueError(f"unknown motor populations: {unknown}")
    selected = {neuron_id for name in names for neuron_id in populations[name]}
    return np.fromiter(
        (int(neuron_id) in selected for neuron_id in graph.neuron_ids),
        dtype=np.bool_,
        count=graph.neuron_count,
    )
