"""Environment-observation metrics and conservative control comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.behavioral_controls import CONDITIONS, ControlCondition


class EpisodeObservation(BaseModel, frozen=True):
    """One episode's measured outcome; no target or neural command fields."""

    model_config = ConfigDict(extra="forbid")

    evaluation_target: Literal["food", "threat", "both"] = "both"
    food_contact: bool
    threat_contact: bool
    initial_food_distance: float = Field(ge=0.0)
    final_food_distance: float = Field(ge=0.0)
    initial_threat_distance: float = Field(ge=0.0)
    final_threat_distance: float = Field(ge=0.0)
    first_food_contact_step: int | None = Field(default=None, ge=0)
    first_threat_contact_step: int | None = Field(default=None, ge=0)
    time_to_clear_threat_steps: int | None = Field(default=None, ge=0)
    horizon_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_finite(self) -> EpisodeObservation:
        if not all(
            math.isfinite(value)
            for value in (
                self.initial_food_distance,
                self.final_food_distance,
                self.initial_threat_distance,
                self.final_threat_distance,
            )
        ):
            raise ValueError("episode distances must be finite")
        return self


class FoodMetrics(BaseModel, frozen=True):
    contact_rate: float = Field(ge=0.0, le=1.0)
    mean_time_to_contact_steps: float = Field(ge=0.0)
    mean_distance_improvement: float


class ThreatMetrics(BaseModel, frozen=True):
    avoidance_rate: float = Field(ge=0.0, le=1.0)
    contact_rate: float = Field(ge=0.0, le=1.0)
    mean_time_to_clear_steps: float = Field(ge=0.0)
    mean_distance_improvement: float


class BootstrapInterval(BaseModel, frozen=True):
    mean: float
    low: float
    high: float
    samples: int = Field(gt=0)


class PairedComparison(BaseModel, frozen=True):
    control: ControlCondition
    food_delta: BootstrapInterval
    threat_delta: BootstrapInterval
    paired_observations: int = Field(gt=0)


def _require_observations(observations: Sequence[EpisodeObservation]) -> None:
    if not observations:
        raise ValueError("at least one episode observation is required")


def food_metrics(observations: Sequence[EpisodeObservation]) -> FoodMetrics:
    _require_observations(observations)
    times = [
        float(item.first_food_contact_step)
        if item.first_food_contact_step is not None
        else float(item.horizon_steps + 1)
        for item in observations
    ]
    return FoodMetrics(
        contact_rate=sum(item.food_contact for item in observations) / len(observations),
        mean_time_to_contact_steps=float(np.mean(times)),
        mean_distance_improvement=float(
            np.mean(
                [item.initial_food_distance - item.final_food_distance for item in observations]
            )
        ),
    )


def threat_metrics(observations: Sequence[EpisodeObservation]) -> ThreatMetrics:
    _require_observations(observations)
    times = [
        float(item.time_to_clear_threat_steps)
        if item.time_to_clear_threat_steps is not None
        else float(item.horizon_steps + 1)
        for item in observations
    ]
    return ThreatMetrics(
        avoidance_rate=sum(not item.threat_contact for item in observations)
        / len(observations),
        contact_rate=sum(item.threat_contact for item in observations) / len(observations),
        mean_time_to_clear_steps=float(np.mean(times)),
        mean_distance_improvement=float(
            np.mean(
                [
                    item.final_threat_distance - item.initial_threat_distance
                    for item in observations
                ]
            )
        ),
    )


def bootstrap_interval(
    values: Sequence[float], *, seed: int, samples: int = 2_000
) -> BootstrapInterval:
    if not values or samples <= 0:
        raise ValueError("bootstrap requires nonempty values and positive samples")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return BootstrapInterval(
        mean=float(array.mean()), low=float(low), high=float(high), samples=samples
    )


def compare_conditions(
    normal: Sequence[EpisodeObservation],
    control_observations: Sequence[EpisodeObservation],
    *,
    control: ControlCondition,
    seed: int,
    samples: int = 2_000,
    replicate_size: int = 1,
    food_replicate_size: int | None = None,
    threat_replicate_size: int | None = None,
) -> PairedComparison:
    """Compare paired environment outcomes without exposing task state to the agent."""

    if len(normal) != len(control_observations) or not normal:
        raise ValueError("condition comparison requires nonempty paired observations")
    if len(normal) != len(control_observations):
        raise ValueError("condition comparison requires equal paired observations")
    food_size = food_replicate_size or replicate_size
    threat_size = threat_replicate_size or replicate_size
    if food_size <= 0 or threat_size <= 0:
        raise ValueError("replicate sizes must be positive")
    paired = tuple(zip(normal, control_observations, strict=True))
    for normal_item, control_item in paired:
        if normal_item.evaluation_target != control_item.evaluation_target:
            raise ValueError("paired observations must declare the same evaluation target")
    food_pairs = tuple(
        (normal_item, control_item)
        for normal_item, control_item in paired
        if normal_item.evaluation_target in {"food", "both"}
    )
    threat_pairs = tuple(
        (normal_item, control_item)
        for normal_item, control_item in paired
        if normal_item.evaluation_target in {"threat", "both"}
    )
    if not food_pairs or not threat_pairs:
        raise ValueError("condition comparison requires both food and threat holdouts")
    food_values = tuple(
        (normal_item.food_contact - control_item.food_contact)
        + (
            (normal_item.initial_food_distance - normal_item.final_food_distance)
            - (control_item.initial_food_distance - control_item.final_food_distance)
        )
        for normal_item, control_item in food_pairs
    )
    threat_values = tuple(
        ((not normal_item.threat_contact) - (not control_item.threat_contact))
        + (
            (normal_item.final_threat_distance - normal_item.initial_threat_distance)
            - (
                control_item.final_threat_distance
                - control_item.initial_threat_distance
            )
        )
        for normal_item, control_item in threat_pairs
    )
    if len(food_values) % food_size or len(threat_values) % threat_size:
        raise ValueError("task-specific replicate sizes must evenly partition observations")
    food_replicates = tuple(
        float(np.mean(food_values[start : start + food_size]))
        for start in range(0, len(food_values), food_size)
    )
    threat_replicates = tuple(
        float(np.mean(threat_values[start : start + threat_size]))
        for start in range(0, len(threat_values), threat_size)
    )
    return PairedComparison(
        control=control,
        food_delta=bootstrap_interval(food_replicates, seed=seed, samples=samples),
        threat_delta=bootstrap_interval(threat_replicates, seed=seed + 1, samples=samples),
        paired_observations=min(len(food_replicates), len(threat_replicates)),
    )


def claim_gate(
    comparisons: Mapping[ControlCondition, PairedComparison],
    *,
    replay_exact: bool = True,
    graph_unchanged: bool = True,
    generalization_verified: bool = False,
) -> bool:
    """Allow a behavioral claim only with causal, reproducible evidence."""

    if not replay_exact or not graph_unchanged or not generalization_verified:
        return False
    required = set(CONDITIONS[1:])
    if set(comparisons) != required:
        return False
    return all(
        math.isfinite(comparison.food_delta.mean)
        and math.isfinite(comparison.threat_delta.mean)
        and comparison.food_delta.low > 0.0
        and comparison.threat_delta.low > 0.0
        and comparison.food_delta.samples > 1
        and comparison.threat_delta.samples > 1
        and comparison.paired_observations >= 3
        for comparison in comparisons.values()
    )
