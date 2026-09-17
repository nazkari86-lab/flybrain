"""Predeclared anonymous sensory schedules for conditioning experiments."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConditioningEvent(BaseModel, frozen=True):
    """One anonymous sensory state delivered at a fixed timestep."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    odor_intensity: float = Field(ge=0.0, le=1.0)
    appetitive_contact_intensity: float = Field(ge=0.0, le=1.0)
    aversive_contact_intensity: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_one_contact_valence(self) -> ConditioningEvent:
        if self.appetitive_contact_intensity and self.aversive_contact_intensity:
            raise ValueError("conditioning event cannot contain both contact valences")
        return self


class ConditioningSchedule(BaseModel, frozen=True):
    """A deterministic stimulus sequence fully chosen before neural execution."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    events: tuple[ConditioningEvent, ...]

    @model_validator(mode="after")
    def validate_order(self) -> ConditioningSchedule:
        expected = tuple(range(len(self.events)))
        if tuple(event.step for event in self.events) != expected:
            raise ValueError("conditioning schedule steps must be contiguous and ordered")
        return self

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        steps: int,
        appetitive_pair_steps: tuple[int, ...] = (),
        aversive_pair_steps: tuple[int, ...] = (),
    ) -> ConditioningSchedule:
        """Build an immutable schedule without reading neural state or body actions."""

        if steps <= 0:
            raise ValueError("conditioning steps must be positive")
        appetitive = frozenset(appetitive_pair_steps)
        aversive = frozenset(aversive_pair_steps)
        if appetitive & aversive:
            raise ValueError("conditioning pair steps cannot contain both valences")
        invalid = sorted((appetitive | aversive) - set(range(steps)))
        if invalid:
            raise ValueError(f"conditioning pair step outside schedule: {invalid}")
        return cls(
            seed=seed,
            events=tuple(
                ConditioningEvent(
                    step=step,
                    odor_intensity=1.0 if step in appetitive | aversive else 0.0,
                    appetitive_contact_intensity=1.0 if step in appetitive else 0.0,
                    aversive_contact_intensity=1.0 if step in aversive else 0.0,
                )
                for step in range(steps)
            ),
        )

    @property
    def digest(self) -> str:
        """Stable digest proves an executed schedule was fixed before the episode."""

        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
