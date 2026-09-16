"""Evidence-bound descending-neuron summaries and causal silence masks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flybrain.embodied_world import MotorCommand
from flybrain.graph import EventConnectome


def _ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique IDs")
    return values


@dataclass(frozen=True)
class DescendingMap:
    """Resolved side-specific populations used by the bounded decoder."""

    d_na02_left: tuple[int, ...]
    d_na02_right: tuple[int, ...]
    d_ng13_left: tuple[int, ...]
    d_ng13_right: tuple[int, ...]
    mdn_left: tuple[int, ...]
    mdn_right: tuple[int, ...]

    def __post_init__(self) -> None:
        names = (
            "d_na02_left",
            "d_na02_right",
            "d_ng13_left",
            "d_ng13_right",
            "mdn_left",
            "mdn_right",
        )
        groups = tuple(_ids(getattr(self, name), name) for name in names)
        flattened = [value for group in groups for value in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("descending populations must be disjoint")

    def named_populations(self) -> dict[str, tuple[int, ...]]:
        return {
            name: getattr(self, name)
            for name in (
                "d_na02_left",
                "d_na02_right",
                "d_ng13_left",
                "d_ng13_right",
                "mdn_left",
                "mdn_right",
            )
        }


@dataclass(frozen=True)
class DescendingActivity:
    """Normalized population rates and the resulting target-independent command."""

    command: MotorCommand
    d_na02_left_rate: float
    d_na02_right_rate: float
    d_ng13_left_rate: float
    d_ng13_right_rate: float
    mdn_left_rate: float
    mdn_right_rate: float


class DescendingDecoder:
    """Decode only declared DN spikes into bounded steering and retreat."""

    version = "evidence-dn-v1"

    def __init__(self, mapping: DescendingMap, *, walking_drive: float = 0.2) -> None:
        if not np.isfinite(walking_drive) or not -1.0 <= walking_drive <= 1.0:
            raise ValueError("walking_drive must be finite and between -1 and 1")
        self.mapping = mapping
        self.walking_drive = float(walking_drive)

    def decode(self, spikes: tuple[int, ...]) -> DescendingActivity:
        counts = Counter(spikes)

        def rate(ids: tuple[int, ...]) -> float:
            return sum(counts[value] for value in ids) / len(ids)

        na_l, na_r = rate(self.mapping.d_na02_left), rate(self.mapping.d_na02_right)
        ng_l, ng_r = rate(self.mapping.d_ng13_left), rate(self.mapping.d_ng13_right)
        mdn_l, mdn_r = rate(self.mapping.mdn_left), rate(self.mapping.mdn_right)
        steering = 0.5 * ((na_l - na_r) + (ng_l - ng_r))
        retreat = 0.5 * (mdn_l + mdn_r)
        retreat_turn = 0.25 * (mdn_l - mdn_r)
        return DescendingActivity(
            command=MotorCommand(
                forward=float(np.clip(self.walking_drive - 2.0 * retreat, -1.0, 1.0)),
                turn=float(np.clip(steering + retreat_turn, -1.0, 1.0)),
            ),
            d_na02_left_rate=na_l,
            d_na02_right_rate=na_r,
            d_ng13_left_rate=ng_l,
            d_ng13_right_rate=ng_r,
            mdn_left_rate=mdn_l,
            mdn_right_rate=mdn_r,
        )


def population_silence_mask(
    graph: EventConnectome,
    mapping: DescendingMap,
    names: frozenset[str],
) -> NDArray[np.bool_]:
    """Return an index-aligned mask for exactly the named DN populations."""

    populations = mapping.named_populations()
    unknown = sorted(names - populations.keys())
    if unknown:
        raise ValueError(f"unknown descending populations: {unknown}")
    selected = {value for name in names for value in populations[name]}
    if names and not selected:
        raise ValueError("requested descending population is empty")
    return np.fromiter(
        (int(neuron_id) in selected for neuron_id in graph.neuron_ids),
        dtype=np.bool_,
        count=graph.neuron_count,
    )
