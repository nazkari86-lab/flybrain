"""Perturbation-recovery checks for sparse connectome dynamics."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.graph import EventConnectome
from flybrain.shiu import (
    ShiuParameters,
    ShiuState,
    poisson_voltage_events,
    simulate_shiu,
)


class PerturbationRecoveryAudit(BaseModel, frozen=True):
    """Simulation observation of whether a finite sensory pulse settles."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    stimulus_mode: Literal["direct_voltage", "source_poisson"]
    classification: Literal["recovered", "persistent_activity", "underpowered"]
    stimulus_voltage_events: int = Field(ge=0)
    window_spikes: tuple[int, ...]
    tail_to_initial_fraction: float = Field(ge=0.0)
    replay_exact: bool
    graph_unchanged: bool
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for values in (
        graph.neuron_ids,
        graph.outgoing.data,
        graph.outgoing.indices,
        graph.outgoing.indptr,
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _indices(graph: EventConnectome, neuron_ids: tuple[int, ...], name: str) -> np.ndarray:
    if not neuron_ids or any(type(item) is not int or item <= 0 for item in neuron_ids):
        raise ValueError(f"{name} IDs must be nonempty positive integers")
    if len(set(neuron_ids)) != len(neuron_ids):
        raise ValueError(f"{name} IDs must be unique")
    index_by_id = {int(neuron_id): index for index, neuron_id in enumerate(graph.neuron_ids)}
    try:
        return np.asarray([index_by_id[item] for item in neuron_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{name} ID absent from graph: {error.args[0]}") from error


def audit_perturbation_recovery(
    graph: EventConnectome,
    *,
    stimulus_ids: tuple[int, ...],
    observed_ids: tuple[int, ...],
    params: ShiuParameters,
    stimulus_voltage_mv: float,
    window_steps: int,
    recovery_windows: int,
    stimulus_mode: Literal["direct_voltage", "source_poisson"] = "direct_voltage",
    maximum_tail_fraction: float = 0.1,
    seed: int = 7,
) -> PerturbationRecoveryAudit:
    """Deliver one finite pulse and audit observed activity in following windows."""

    if type(window_steps) is not int or window_steps <= 0:
        raise ValueError("window_steps must be a positive integer")
    if type(recovery_windows) is not int or recovery_windows <= 0:
        raise ValueError("recovery_windows must be a positive integer")
    if not np.isfinite(stimulus_voltage_mv) or stimulus_voltage_mv <= 0.0:
        raise ValueError("stimulus voltage must be finite and positive")
    if not np.isfinite(maximum_tail_fraction) or not 0.0 <= maximum_tail_fraction <= 1.0:
        raise ValueError("maximum tail fraction must lie in [0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    params.validate()
    before = _graph_digest(graph)
    stimulus_indices = _indices(graph, stimulus_ids, "stimulus")
    observed_indices = _indices(graph, observed_ids, "observed")
    observed_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
    observed_mask[observed_indices] = True
    total_steps = window_steps * (recovery_windows + 1)
    event = (
        {
            0: (
                stimulus_indices,
                np.full(
                    stimulus_indices.size,
                    stimulus_voltage_mv,
                    dtype=np.float32,
                ),
            )
        }
        if stimulus_mode == "direct_voltage"
        else poisson_voltage_events(
            stimulus_indices,
            steps=window_steps,
            params=params,
            seed=seed,
        )
    )

    def run() -> tuple[tuple[int, ...], str]:
        state = ShiuState.initial(graph.neuron_count, params=params, seed=seed)
        counts = [0] * (recovery_windows + 1)
        digest = hashlib.sha256()
        for batch in simulate_shiu(
            graph,
            params,
            steps=total_steps,
            external_voltage_events=event,
            state=state,
        ):
            batch_indices = np.searchsorted(graph.neuron_ids, batch.neuron_ids)
            observed = np.flatnonzero(observed_mask[batch_indices])
            counts[batch.step // window_steps] += int(observed.size)
            digest.update(batch.step.to_bytes(8, byteorder="little", signed=False))
            digest.update(batch.neuron_ids.tobytes())
        return tuple(counts), digest.hexdigest()

    first_counts, first_digest = run()
    replay_counts, replay_digest = run()
    initial = first_counts[0]
    tail = max(first_counts[1:])
    fraction = (
        0.0
        if initial == 0 and tail == 0
        else (float("inf") if initial == 0 else tail / initial)
    )
    if initial == 0:
        classification: Literal["recovered", "persistent_activity", "underpowered"] = (
            "underpowered"
        )
    elif fraction > maximum_tail_fraction:
        classification = "persistent_activity"
    else:
        classification = "recovered"
    return PerturbationRecoveryAudit(
        stimulus_mode=stimulus_mode,
        classification=classification,
        stimulus_voltage_events=sum(int(indices.size) for indices, _ in event.values()),
        window_spikes=first_counts,
        tail_to_initial_fraction=0.0 if not np.isfinite(fraction) else fraction,
        replay_exact=first_counts == replay_counts and first_digest == replay_digest,
        graph_unchanged=_graph_digest(graph) == before,
        trace_digest=first_digest,
    )
