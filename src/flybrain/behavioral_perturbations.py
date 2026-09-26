"""Bounded physical perturbations used by autonomous behavior benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.hexapod_body import HexapodParameters


class BodyPerturbation(BaseModel, frozen=True):
    """An environment perturbation that never becomes a neural command."""

    model_config = ConfigDict(extra="forbid")

    friction_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    mass_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    delay_steps: int = Field(default=0, ge=0, le=100)
    damaged_legs: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_legs(self) -> Self:
        if any(index < 0 or index >= 6 for index in self.damaged_legs):
            raise ValueError("damaged leg indices must be between 0 and 5")
        if len(set(self.damaged_legs)) != len(self.damaged_legs):
            raise ValueError("damaged leg indices must be unique")
        return self


class BehaviorVariant(BaseModel, frozen=True):
    """One deterministic world variant used for training or holdout evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    evaluation_target: Literal["food", "threat", "both"] = "both"
    food_position_m: tuple[float, float]
    threat_position_m: tuple[float, float]
    initial_position_m: tuple[float, float] = (0.0, 0.0)
    perturbation: BodyPerturbation = BodyPerturbation()

    @model_validator(mode="after")
    def validate_positions(self) -> Self:
        values = (*self.food_position_m, *self.threat_position_m, *self.initial_position_m)
        if any(not isinstance(value, (int, float)) for value in values):
            raise ValueError("behavior positions must be numeric")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("behavior positions must be finite")
        return self


def world_digest(variant: BehaviorVariant) -> str:
    """Return a stable identity for geometry and physics, excluding score labels."""

    payload = variant.model_dump(
        mode="json", exclude={"name", "evaluation_target"}
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scale_behavior_variant(variant: BehaviorVariant, scale: float) -> BehaviorVariant:
    """Rescale environment geometry without changing labels or body physics."""

    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("arena scale must be finite and in (0, 1]")
    return variant.model_copy(
        update={
            "food_position_m": tuple(value * scale for value in variant.food_position_m),
            "threat_position_m": tuple(value * scale for value in variant.threat_position_m),
            "initial_position_m": tuple(value * scale for value in variant.initial_position_m),
        }
    )


def apply_perturbation(
    parameters: HexapodParameters,
    perturbation: BodyPerturbation,
) -> HexapodParameters:
    """Return perturbed body parameters while preserving the neural timestep."""

    return parameters.model_copy(
        update={
            "friction_coefficient": (
                parameters.friction_coefficient * perturbation.friction_scale
            ),
            "body_mass_kg": parameters.body_mass_kg * perturbation.mass_scale,
        }
    )
