"""Anonymous contact-to-DAN recruitment with no privileged task-state channel."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.biological_registry import ResolvedRegistry


class AnonymousContact(BaseModel, frozen=True):
    """Bounded physical-contact features, deliberately without world-object identity."""

    model_config = ConfigDict(extra="forbid")

    appetitive_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    aversive_intensity: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_single_valence(self) -> AnonymousContact:
        if not all(math.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("contact intensities must be finite")
        if self.appetitive_intensity and self.aversive_intensity:
            raise ValueError("a contact cannot recruit both DAN valences")
        return self


class DANRecruitment(BaseModel, frozen=True):
    """A declared physical-contact recruitment record, not a reward scalar."""

    dan_ids: tuple[int, ...]
    intensity: float = Field(ge=0.0, le=1.0)


class ReinforcementInterface(BaseModel, frozen=True):
    """Route only declared appetitive or aversive contact channels to DAN populations."""

    model_config = ConfigDict(extra="forbid")

    appetitive_dan_ids: tuple[int, ...]
    aversive_dan_ids: tuple[int, ...]

    @classmethod
    def from_resolved_registry(cls, registry: ResolvedRegistry) -> ReinforcementInterface:
        """Bind only exact registry roles; unassigned DAN types cannot enter this interface."""

        by_role = {
            role: tuple(
                neuron_id
                for population in registry.populations
                if population.role == role
                for neuron_id in population.neuron_ids
            )
            for role in ("dan_appetitive", "dan_aversive")
        }
        return cls(
            appetitive_dan_ids=tuple(sorted(by_role["dan_appetitive"])),
            aversive_dan_ids=tuple(sorted(by_role["dan_aversive"])),
        )

    @model_validator(mode="after")
    def validate_dan_populations(self) -> ReinforcementInterface:
        for name, ids in (
            ("appetitive", self.appetitive_dan_ids),
            ("aversive", self.aversive_dan_ids),
        ):
            if not ids or any(type(neuron_id) is not int or neuron_id <= 0 for neuron_id in ids):
                raise ValueError(f"{name} DAN IDs must be positive and nonempty")
            if tuple(sorted(set(ids))) != ids:
                raise ValueError(f"{name} DAN IDs must be unique and sorted")
        if set(self.appetitive_dan_ids) & set(self.aversive_dan_ids):
            raise ValueError("appetitive and aversive DAN populations overlap")
        return self

    def recruit(self, contact: AnonymousContact) -> DANRecruitment:
        """Map one bounded contact feature to exactly one permitted DAN group or silence."""

        if contact.appetitive_intensity:
            return DANRecruitment(
                dan_ids=self.appetitive_dan_ids,
                intensity=contact.appetitive_intensity,
            )
        if contact.aversive_intensity:
            return DANRecruitment(
                dan_ids=self.aversive_dan_ids,
                intensity=contact.aversive_intensity,
            )
        return DANRecruitment(dan_ids=(), intensity=0.0)
