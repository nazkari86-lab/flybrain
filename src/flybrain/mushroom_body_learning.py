"""Local, bounded three-factor learning on a sparse KC-to-MBON overlay."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import cast

import numpy as np
from numpy.typing import NDArray

from flybrain.plastic_overlay import PlasticWeightOverlay


@dataclass(frozen=True)
class MushroomBodyLearningParameters:
    """Predeclared model assumptions for local eligibility and depression."""

    eligibility_tau_ms: float = 1_000.0
    learning_rate: float = 0.05
    minimum_multiplier: float = 0.2
    maximum_multiplier: float = 1.0

    def validate(self) -> None:
        values = (
            self.eligibility_tau_ms,
            self.learning_rate,
            self.minimum_multiplier,
            self.maximum_multiplier,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("learning parameters must be finite")
        if self.eligibility_tau_ms <= 0 or self.learning_rate < 0:
            raise ValueError("learning time constant and rate are invalid")
        if not 0 <= self.minimum_multiplier <= 1 <= self.maximum_multiplier:
            raise ValueError("learning multiplier bounds must contain one")


@dataclass
class MushroomBodyLearning:
    """Eligibility exists only for coincident declared KC and MBON edge endpoints."""

    overlay: PlasticWeightOverlay
    edge_pre_ids: NDArray[np.uint64]
    edge_post_ids: NDArray[np.uint64]
    dan_ids: NDArray[np.uint64]
    dan_post_ids: NDArray[np.uint64]
    parameters: MushroomBodyLearningParameters
    eligibility: NDArray[np.float32]

    def __init__(
        self,
        *,
        overlay: PlasticWeightOverlay,
        edge_pre_ids: NDArray[np.uint64],
        edge_post_ids: NDArray[np.uint64],
        dan_ids: NDArray[np.uint64],
        dan_post_ids: NDArray[np.uint64],
        parameters: MushroomBodyLearningParameters,
    ) -> None:
        parameters.validate()
        pre_ids = edge_pre_ids.astype(np.uint64, copy=True)
        post_ids = edge_post_ids.astype(np.uint64, copy=True)
        route_dan_ids = dan_ids.astype(np.uint64, copy=True)
        route_post_ids = dan_post_ids.astype(np.uint64, copy=True)
        if any(
            values.ndim != 1
            for values in (pre_ids, post_ids, route_dan_ids, route_post_ids)
        ):
            raise ValueError("plastic edge endpoints must be one-dimensional")
        if pre_ids.shape != post_ids.shape or pre_ids.shape != overlay.multipliers.shape:
            raise ValueError("plastic edge endpoints differ from overlay manifest")
        if route_dan_ids.shape != route_post_ids.shape:
            raise ValueError("DAN routing endpoints have different shapes")
        self.overlay = overlay
        self.edge_pre_ids = pre_ids
        self.edge_post_ids = post_ids
        self.dan_ids = route_dan_ids
        self.dan_post_ids = route_post_ids
        self.parameters = parameters
        self.eligibility = np.zeros(pre_ids.size, dtype=np.float32)

    def step(
        self,
        *,
        active_kc_ids: NDArray[np.uint64],
        active_mbon_ids: NDArray[np.uint64],
        routed_dan_ids: NDArray[np.uint64],
        dt_ms: float,
    ) -> None:
        """Decay local traces, record coactivity, then apply a routed DAN third factor."""

        if not isfinite(dt_ms) or dt_ms < 0:
            raise ValueError("learning timestep must be finite and non-negative")
        for event_ids in (active_kc_ids, active_mbon_ids, routed_dan_ids):
            if event_ids.ndim != 1:
                raise ValueError("learning event IDs must be one-dimensional")
        self.eligibility *= np.float32(exp(-dt_ms / self.parameters.eligibility_tau_ms))
        eligible = np.isin(self.edge_pre_ids, active_kc_ids) & np.isin(
            self.edge_post_ids, active_mbon_ids
        )
        self.eligibility[eligible] += 1.0
        if routed_dan_ids.size:
            reachable_posts = self.dan_post_ids[
                np.isin(self.dan_ids, routed_dan_ids)
            ]
            routed_eligibility = self.eligibility * np.isin(
                self.edge_post_ids, reachable_posts
            ).astype(np.float32)
            values = cast(
                NDArray[np.float32],
                np.asarray(
                    self.overlay.multipliers
                    - (np.float32(self.parameters.learning_rate) * routed_eligibility),
                    dtype=np.float32,
                ),
            )
            np.clip(
                values,
                self.parameters.minimum_multiplier,
                self.parameters.maximum_multiplier,
                out=values,
            )
            self.overlay.set_multipliers(
                values,
                minimum=self.parameters.minimum_multiplier,
                maximum=self.parameters.maximum_multiplier,
            )
