"""Predeclared gain sweeps that preserve connectome topology and signs."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from flybrain.dynamics_stability_audit import (
    PerturbationRecoveryAudit,
    audit_perturbation_recovery,
)
from flybrain.graph import EventConnectome
from flybrain.shiu import ShiuParameters


class SynapticGainSweepPoint(BaseModel, frozen=True):
    """One finite-dynamics observation at an explicitly assumed global gain."""

    model_config = ConfigDict(extra="forbid")

    synapse_mv: float = Field(gt=0.0)
    audit: PerturbationRecoveryAudit
    accepted: bool


class SynapticGainSweep(BaseModel, frozen=True):
    """A reproducible selection, never a measurement of biological synaptic gain."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    selection_kind: Literal["model_assumption"] = "model_assumption"
    acceptance_rule: str
    points: tuple[SynapticGainSweepPoint, ...]
    selected_synapse_mv: float | None


def sweep_synaptic_gain(
    graph: EventConnectome,
    *,
    stimulus_ids: tuple[int, ...],
    observed_ids: tuple[int, ...],
    params: ShiuParameters,
    candidate_synapse_mv: tuple[float, ...],
    stimulus_voltage_mv: float,
    window_steps: int,
    recovery_windows: int,
    stimulus_mode: Literal["direct_voltage", "source_poisson"] = "source_poisson",
    maximum_tail_fraction: float = 0.1,
    seed: int = 7,
) -> SynapticGainSweep:
    """Select the first candidate with a response, recovery, exact replay, and no graph change."""

    if not candidate_synapse_mv:
        raise ValueError("candidate synaptic gains must be nonempty")
    if any(gain <= 0.0 for gain in candidate_synapse_mv):
        raise ValueError("candidate synaptic gains must be positive")
    if tuple(sorted(candidate_synapse_mv)) != candidate_synapse_mv:
        raise ValueError("candidate synaptic gains must be strictly ascending")
    if len(set(candidate_synapse_mv)) != len(candidate_synapse_mv):
        raise ValueError("candidate synaptic gains must be unique")

    points: list[SynapticGainSweepPoint] = []
    selected: float | None = None
    for gain in candidate_synapse_mv:
        audit = audit_perturbation_recovery(
            graph,
            stimulus_ids=stimulus_ids,
            observed_ids=observed_ids,
            params=replace(params, synapse_mv=gain),
            stimulus_voltage_mv=stimulus_voltage_mv,
            window_steps=window_steps,
            recovery_windows=recovery_windows,
            stimulus_mode=stimulus_mode,
            maximum_tail_fraction=maximum_tail_fraction,
            seed=seed,
        )
        accepted = (
            audit.classification == "recovered"
            and max(audit.window_spikes) > 0
            and audit.replay_exact
            and audit.graph_unchanged
        )
        points.append(SynapticGainSweepPoint(synapse_mv=gain, audit=audit, accepted=accepted))
        if accepted and selected is None:
            selected = gain
    return SynapticGainSweep(
        acceptance_rule=(
            "nonzero observed response; recovered tail-to-peak activity; "
            "exact replay; immutable graph"
        ),
        points=tuple(points),
        selected_synapse_mv=selected,
    )
