"""Deterministic reference leaky integrate-and-fire dynamics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flybrain.graph import SparseConnectome


@dataclass(frozen=True)
class LIFParameters:
    """Shared parameters for the initial reference neuron population."""

    dt_ms: float
    tau_ms: float
    rest_mv: float
    reset_mv: float
    threshold_mv: float


@dataclass
class LIFState:
    """Complete mutable state required to continue a simulation."""

    voltage: NDArray[np.float32]
    recurrent_current: NDArray[np.float32]
    step: int
    rng_state: Mapping[str, Any]

    @classmethod
    def initial(cls, neuron_count: int, seed: int) -> LIFState:
        generator = np.random.default_rng(seed)
        return cls(
            voltage=np.zeros(neuron_count, dtype=np.float32),
            recurrent_current=np.zeros(neuron_count, dtype=np.float32),
            step=0,
            rng_state=generator.bit_generator.state,
        )


@dataclass(frozen=True)
class SpikeBatch:
    """Neuron IDs emitted at one discrete simulation step."""

    step: int
    neuron_ids: NDArray[np.uint64]


def simulate_lif(
    graph: SparseConnectome,
    params: LIFParameters,
    input_currents: Iterable[NDArray[np.float32]],
    *,
    seed: int,
    state: LIFState | None = None,
    silenced: NDArray[np.bool_] | None = None,
) -> Iterator[SpikeBatch]:
    """Advance shared-parameter LIF neurons using one-step recurrent delays."""

    active_state = state if state is not None else LIFState.initial(graph.neuron_count, seed)
    silence_mask = (
        silenced
        if silenced is not None
        else np.zeros(graph.neuron_count, dtype=np.bool_)
    )
    expected_shape = (graph.neuron_count,)
    if active_state.voltage.shape != expected_shape or silence_mask.shape != expected_shape:
        raise ValueError("state and silence mask must match graph neuron count")

    integration_scale = np.float32(params.dt_ms / params.tau_ms)
    for external_current in input_currents:
        if external_current.shape != expected_shape:
            message = f"expected current shape {expected_shape}, got {external_current.shape}"
            raise ValueError(message)

        total_current = external_current + active_state.recurrent_current
        active_state.voltage += (
            -(active_state.voltage - params.rest_mv) + total_current
        ) * integration_scale
        fired = active_state.voltage >= params.threshold_mv
        fired &= ~silence_mask
        fired_ids = graph.neuron_ids[fired].copy()
        active_state.voltage[fired] = params.reset_mv
        active_state.recurrent_current = graph.propagate(fired.astype(np.float32))
        step = active_state.step
        active_state.step += 1
        yield SpikeBatch(step=step, neuron_ids=fired_ids)
