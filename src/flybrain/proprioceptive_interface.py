"""Privileged-state-free hexapod proprioceptive observation and neural encoding."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flybrain.biological_registry import ResolvedRegistry
from flybrain.embodied_interfaces import ExternalEvent
from flybrain.graph import EventConnectome
from flybrain.hexapod_body import LEG_NAMES, HexapodBody, HexapodParameters, LegName, Vec3


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("proprioceptive value must be finite")
    return max(0.0, min(1.0, value))


class ProprioceptiveCalibration(BaseModel, frozen=True):
    """Explicit transfer-function assumptions for body-to-neuron encoding."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["model_assumption"] = "model_assumption"
    velocity_scale_rad_s: float = Field(default=10.0, gt=0.0)
    load_scale_n: float = Field(default=0.001, gt=0.0)
    total_voltage: float = Field(default=68.75, gt=0.0)
    aggregation: Literal["equal_weight_mean"] = "equal_weight_mean"

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if any(
            not math.isfinite(value)
            for value in (
                self.velocity_scale_rad_s,
                self.load_scale_n,
                self.total_voltage,
            )
        ):
            raise ValueError("proprioceptive calibration must be finite")
        return self


class ProprioceptiveLegObservation(BaseModel, frozen=True):
    """Eight bounded present-state channels for one leg."""

    model_config = ConfigDict(extra="forbid")

    leg: LegName
    joint_angles: Vec3
    joint_velocities: Vec3
    contact: float = Field(ge=0.0, le=1.0)
    load: float = Field(ge=0.0, le=1.0)
    drive: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_channels(self) -> Self:
        channels = (
            *self.joint_angles,
            *self.joint_velocities,
            self.contact,
            self.load,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in channels):
            raise ValueError("proprioceptive channels must be finite values in [0, 1]")
        expected_drive = sum(channels) / len(channels)
        if not math.isclose(self.drive, expected_drive, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("proprioceptive drive disagrees with declared aggregation")
        return self


class ProprioceptiveObservation(BaseModel, frozen=True):
    """Canonical six-leg observation with no world, target, or action state."""

    model_config = ConfigDict(extra="forbid")

    legs: tuple[ProprioceptiveLegObservation, ...]
    calibration: ProprioceptiveCalibration

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        names = tuple(item.leg for item in self.legs)
        if names != LEG_NAMES:
            raise ValueError(f"proprioceptive legs must follow canonical order: {LEG_NAMES}")
        return self

    def leg(self, name: LegName) -> ProprioceptiveLegObservation:
        return self.legs[LEG_NAMES.index(name)]


class ProprioceptiveBank(BaseModel, frozen=True):
    """One exact leg-specific registry population."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    leg: LegName
    neuron_ids: tuple[int, ...]

    @field_validator("neuron_ids", mode="before")
    @classmethod
    def validate_id_types(cls, value: object) -> object:
        if not isinstance(value, (tuple, list)) or any(
            type(item) is not int for item in value
        ):
            raise ValueError("proprioceptive bank must contain integer IDs")
        return value

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        if self.name != f"{self.leg}_proprioception":
            raise ValueError("proprioceptive bank name does not match its leg")
        if not self.neuron_ids or any(value <= 0 for value in self.neuron_ids):
            raise ValueError("proprioceptive bank must contain positive IDs")
        if tuple(sorted(set(self.neuron_ids))) != self.neuron_ids:
            raise ValueError("proprioceptive bank IDs must be unique and sorted")
        return self


class ProprioceptiveMap(BaseModel, frozen=True):
    """Complete disjoint six-bank proprioceptive map."""

    model_config = ConfigDict(extra="forbid")

    banks: tuple[ProprioceptiveBank, ...]

    @model_validator(mode="after")
    def validate_map(self) -> Self:
        expected = tuple(f"{leg}_proprioception" for leg in LEG_NAMES)
        names = tuple(bank.name for bank in self.banks)
        if names != expected:
            raise ValueError("proprioceptive map must contain the complete canonical bank order")
        owner: dict[int, str] = {}
        for bank in self.banks:
            overlap = sorted(set(bank.neuron_ids) & owner.keys())
            if overlap:
                raise ValueError(f"proprioceptive bank overlap for {bank.name}: {overlap}")
            owner.update(dict.fromkeys(bank.neuron_ids, bank.name))
        return self

    @classmethod
    def from_registry(cls, registry: ResolvedRegistry) -> ProprioceptiveMap:
        banks = []
        for leg in LEG_NAMES:
            name = f"{leg}_proprioception"
            population = registry.population(name)
            if population.role != "sensory":
                raise ValueError(f"resolved population is not sensory: {name}")
            banks.append(
                ProprioceptiveBank(
                    name=name,
                    leg=leg,
                    neuron_ids=population.neuron_ids,
                )
            )
        return cls(banks=tuple(banks))

    def bank(self, leg: LegName) -> ProprioceptiveBank:
        return self.banks[LEG_NAMES.index(leg)]

    def validate_graph(self, graph: EventConnectome) -> None:
        available = {int(value) for value in graph.neuron_ids}
        required = {neuron_id for bank in self.banks for neuron_id in bank.neuron_ids}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"proprioceptive bank IDs absent from graph: {missing}")


def observe_proprioception(
    body: HexapodBody,
    parameters: HexapodParameters,
    calibration: ProprioceptiveCalibration,
) -> ProprioceptiveObservation:
    """Observe only current joints and foot contact/load, normalized to [0, 1]."""

    observations = []
    for leg in body.legs:
        angle_values = tuple(
            _bounded((value - lower) / (upper - lower))
            for value, lower, upper in zip(
                leg.joint_angles_rad,
                parameters.joint_lower_rad,
                parameters.joint_upper_rad,
                strict=True,
            )
        )
        velocity_values = tuple(
            _bounded(0.5 + 0.5 * value / calibration.velocity_scale_rad_s)
            for value in leg.joint_velocities_rad_s
        )
        angles: Vec3 = (angle_values[0], angle_values[1], angle_values[2])
        velocities: Vec3 = (
            velocity_values[0],
            velocity_values[1],
            velocity_values[2],
        )
        contact = 1.0 if leg.contact else 0.0
        load = _bounded(leg.load_n / calibration.load_scale_n)
        channels = (*angles, *velocities, contact, load)
        observations.append(
            ProprioceptiveLegObservation(
                leg=leg.name,
                joint_angles=angles,
                joint_velocities=velocities,
                contact=contact,
                load=load,
                drive=sum(channels) / len(channels),
            )
        )
    return ProprioceptiveObservation(
        legs=tuple(observations),
        calibration=calibration,
    )


class ProprioceptiveEncoder:
    """Encode each leg only into its resolved bank with fixed total authority."""

    version = "hexapod-proprioception-v1"

    def __init__(
        self,
        mapping: ProprioceptiveMap,
        calibration: ProprioceptiveCalibration,
    ) -> None:
        self.mapping = mapping
        self.calibration = calibration

    def encode(
        self,
        observation: ProprioceptiveObservation,
        *,
        step: int,
    ) -> tuple[ExternalEvent, ...]:
        if type(step) is not int or step < 0:
            raise ValueError("proprioceptive event step must be a non-negative integer")
        if observation.calibration != self.calibration:
            raise ValueError("observation calibration differs from encoder calibration")
        events = []
        for item in observation.legs:
            bank = self.mapping.bank(item.leg)
            voltage = self.calibration.total_voltage * item.drive / len(bank.neuron_ids)
            events.append(
                ExternalEvent(
                    step=step,
                    neuron_ids=bank.neuron_ids,
                    voltages=(voltage,) * len(bank.neuron_ids),
                    channel=f"proprioception_{item.leg}",
                )
            )
        return tuple(events)
