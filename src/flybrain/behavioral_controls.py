"""Independent biological control conditions for autonomous behavior assays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay

ControlCondition = Literal[
    "normal",
    "no_plasticity",
    "dan_lesion",
    "kc_mbon_lesion",
    "rewired_control",
]
CONDITIONS: tuple[ControlCondition, ...] = (
    "normal",
    "no_plasticity",
    "dan_lesion",
    "kc_mbon_lesion",
    "rewired_control",
)


@dataclass(frozen=True)
class ConditionBinding:
    """One condition's fully independent mutable state and routing flags."""

    condition: ControlCondition
    overlay: PlasticWeightOverlay
    learning: AssociativeCalibrationConfig
    dan_enabled: bool
    rewired: bool
    pre_ids: tuple[int, ...]
    post_ids: tuple[int, ...]


def build_condition(
    condition: ControlCondition,
    binding: PlasticEdgeBinding,
    learning: AssociativeCalibrationConfig,
    *,
    seed: int,
) -> ConditionBinding:
    """Construct an isolated control without editing canonical graph anatomy."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown control condition: {condition}")
    overlay = binding.overlay.copy()
    if condition == "no_plasticity":
        overlay.reset()
    elif condition == "kc_mbon_lesion":
        overlay.set_multipliers(
            np.zeros_like(overlay.multipliers), minimum=0.0, maximum=2.0
        )
    pre_ids = tuple(int(value) for value in binding.pre_ids)
    post_ids = tuple(int(value) for value in binding.post_ids)
    if condition == "rewired_control":
        generator = np.random.default_rng(seed)
        post_ids = tuple(
            int(value) for value in binding.post_ids[generator.permutation(binding.post_ids.size)]
        )
    return ConditionBinding(
        condition=condition,
        overlay=overlay,
        learning=learning,
        dan_enabled=condition not in {"dan_lesion", "kc_mbon_lesion"},
        rewired=condition == "rewired_control",
        pre_ids=pre_ids,
        post_ids=post_ids,
    )
