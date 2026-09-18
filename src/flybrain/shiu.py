"""Event-driven implementation of the published Shiu et al. LIF model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flybrain.dynamics import SpikeBatch
from flybrain.graph import EventConnectome


@dataclass(frozen=True)
class ShiuParameters:
    """Source-traceable default constants from the Shiu reference model."""

    dt_ms: float = 0.1
    rest_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    conductance_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    synaptic_delay_ms: float = 1.8
    synapse_mv: float = 0.275
    poisson_rate_hz: float = 150.0
    poisson_voltage_scale: float = 250.0

    @property
    def refractory_steps(self) -> int:
        return round(self.refractory_ms / self.dt_ms)

    @property
    def delay_steps(self) -> int:
        return round(self.synaptic_delay_ms / self.dt_ms)

    def validate(self) -> None:
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if self.membrane_tau_ms <= 0 or self.conductance_tau_ms <= 0:
            raise ValueError("time constants must be positive")
        if self.membrane_tau_ms == self.conductance_tau_ms:
            raise ValueError("equal time constants require a separate analytic solution")
        if self.refractory_steps < 1 or self.delay_steps < 1:
            raise ValueError("refractory period and synaptic delay must be at least one step")


@dataclass
class ShiuState:
    """Complete mutable state for deterministic continuation."""

    voltage_mv: NDArray[np.float32]
    conductance_mv: NDArray[np.float32]
    refractory_steps_left: NDArray[np.int32]
    delayed_conductance_mv: NDArray[np.float32]
    step: int
    rng_state: Mapping[str, Any]

    @classmethod
    def initial(
        cls,
        neuron_count: int,
        *,
        params: ShiuParameters,
        seed: int,
    ) -> ShiuState:
        params.validate()
        generator = np.random.default_rng(seed)
        return cls(
            voltage_mv=np.full(neuron_count, params.rest_mv, dtype=np.float32),
            conductance_mv=np.zeros(neuron_count, dtype=np.float32),
            refractory_steps_left=np.zeros(neuron_count, dtype=np.int32),
            delayed_conductance_mv=np.zeros(
                (params.delay_steps + 1, neuron_count), dtype=np.float32
            ),
            step=0,
            rng_state=generator.bit_generator.state,
        )


VoltageEvents = Mapping[int, tuple[NDArray[np.int64], NDArray[np.float32]]]


def poisson_voltage_events(
    neuron_indices: NDArray[np.int64],
    *,
    steps: int,
    params: ShiuParameters,
    seed: int,
) -> dict[int, tuple[NDArray[np.int64], NDArray[np.float32]]]:
    """Generate deterministic source-equivalent Poisson voltage jumps."""

    params.validate()
    if neuron_indices.ndim != 1:
        raise ValueError("neuron_indices must be one-dimensional")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    probability = params.poisson_rate_hz * params.dt_ms / 1000.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Poisson rate and dt imply more than one event per step")

    generator = np.random.default_rng(seed)
    amplitude = np.float32(params.synapse_mv * params.poisson_voltage_scale)
    events: dict[int, tuple[NDArray[np.int64], NDArray[np.float32]]] = {}
    for step in range(steps):
        selected = generator.random(neuron_indices.size) < probability
        if np.any(selected):
            indices = neuron_indices[selected].copy()
            events[step] = (indices, np.full(indices.size, amplitude, dtype=np.float32))
    return events


def _validate_state(
    graph: EventConnectome,
    params: ShiuParameters,
    state: ShiuState,
) -> None:
    vector_shape = (graph.neuron_count,)
    ring_shape = (params.delay_steps + 1, graph.neuron_count)
    if (
        state.voltage_mv.shape != vector_shape
        or state.conductance_mv.shape != vector_shape
        or state.refractory_steps_left.shape != vector_shape
        or state.delayed_conductance_mv.shape != ring_shape
    ):
        raise ValueError("Shiu state dimensions do not match graph and parameters")


def simulate_shiu(
    graph: EventConnectome,
    params: ShiuParameters,
    *,
    steps: int,
    external_voltage_events: VoltageEvents | None = None,
    seed: int = 0,
    state: ShiuState | None = None,
    silenced: NDArray[np.bool_] | None = None,
    refractory_exempt: NDArray[np.bool_] | None = None,
    presynaptic_transmitter_multipliers: Mapping[str, float] | None = None,
) -> Iterator[SpikeBatch]:
    """Advance analytic alpha-synapse LIF dynamics with delayed sparse events."""

    params.validate()
    if steps < 0:
        raise ValueError("steps must be non-negative")
    active_state = (
        state
        if state is not None
        else ShiuState.initial(graph.neuron_count, params=params, seed=seed)
    )
    _validate_state(graph, params, active_state)
    silence_mask = (
        silenced
        if silenced is not None
        else np.zeros(graph.neuron_count, dtype=np.bool_)
    )
    if silence_mask.shape != (graph.neuron_count,):
        raise ValueError("silence mask must match graph neuron count")
    exempt_mask = (
        refractory_exempt
        if refractory_exempt is not None
        else np.zeros(graph.neuron_count, dtype=np.bool_)
    )
    if exempt_mask.shape != (graph.neuron_count,):
        raise ValueError("refractory exemption mask must match graph neuron count")
    voltage_events = external_voltage_events or {}
    transmitter_multipliers = presynaptic_transmitter_multipliers or {}
    if any(
        not isinstance(name, str) or not np.isfinite(value) or value < 0.0
        for name, value in transmitter_multipliers.items()
    ):
        raise ValueError("transmitter multipliers must have string names and finite values >= 0")

    membrane_decay = np.float32(np.exp(-params.dt_ms / params.membrane_tau_ms))
    conductance_decay = np.float32(np.exp(-params.dt_ms / params.conductance_tau_ms))
    coupling = np.float32(
        params.conductance_tau_ms
        / (params.conductance_tau_ms - params.membrane_tau_ms)
        * (conductance_decay - membrane_decay)
    )

    for _ in range(steps):
        step = active_state.step
        ring_index = step % active_state.delayed_conductance_mv.shape[0]
        active_state.conductance_mv += active_state.delayed_conductance_mv[ring_index]
        active_state.delayed_conductance_mv[ring_index].fill(0.0)

        active_state.refractory_steps_left = np.maximum(
            active_state.refractory_steps_left - 1, 0
        )
        refractory = active_state.refractory_steps_left > 0
        active = ~refractory

        prior_conductance = active_state.conductance_mv.copy()
        voltage_delta = active_state.voltage_mv - np.float32(params.rest_mv)
        active_state.voltage_mv[active] = (
            np.float32(params.rest_mv)
            + voltage_delta[active] * membrane_decay
            + prior_conductance[active] * coupling
        )
        active_state.conductance_mv[active] = prior_conductance[active] * conductance_decay
        active_state.voltage_mv[refractory] = np.float32(params.reset_mv)
        active_state.conductance_mv[refractory] = 0.0

        if step in voltage_events:
            indices, amplitudes = voltage_events[step]
            if indices.shape != amplitudes.shape or indices.ndim != 1:
                raise ValueError("external voltage indices and amplitudes must be matching vectors")
            outside_graph = indices.size and (
                int(indices.min()) < 0 or int(indices.max()) >= graph.neuron_count
            )
            if outside_graph:
                raise ValueError("external voltage event targets neuron outside graph")
            eligible = active_state.refractory_steps_left[indices] == 0
            np.add.at(active_state.voltage_mv, indices[eligible], amplitudes[eligible])

        fired = active_state.voltage_mv > params.threshold_mv
        fired &= ~silence_mask
        fired_indices = np.flatnonzero(fired).astype(np.int64, copy=False)
        fired_ids = graph.neuron_ids[fired_indices].copy()
        if fired_indices.size:
            active_state.voltage_mv[fired_indices] = np.float32(params.reset_mv)
            active_state.conductance_mv[fired_indices] = 0.0
            refractory_values = np.where(
                exempt_mask[fired_indices], 0, params.refractory_steps
            ).astype(np.int32, copy=False)
            active_state.refractory_steps_left[fired_indices] = refractory_values
            arrival_step = step + params.delay_steps
            arrival_index = arrival_step % active_state.delayed_conductance_mv.shape[0]
            scales = np.asarray(
                [
                    params.synapse_mv * transmitter_multipliers.get(
                        graph.transmitters[index]
                        if len(graph.transmitters) == graph.neuron_count
                        else "unresolved",
                        1.0,
                    )
                    for index in fired_indices
                ],
                dtype=np.float32,
            )
            active_state.delayed_conductance_mv[arrival_index] += graph.propagate_indices(
                fired_indices, scale=scales
            )

        active_state.step += 1
        yield SpikeBatch(step=step, neuron_ids=fired_ids)
