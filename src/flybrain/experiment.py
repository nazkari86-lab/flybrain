"""Declarative, paired connectome experiment execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import BaseModel, Field

from flybrain.dynamics import LIFParameters, simulate_lif
from flybrain.graph import SparseConnectome
from flybrain.interventions import silence_mask
from flybrain.schema import SnapshotMetadata


class Stimulus(BaseModel, frozen=True):
    """Injected current targeting one canonical neuron at one step."""

    step: int = Field(ge=0)
    neuron_id: int = Field(gt=0)
    current: float


class ExperimentConfig(BaseModel, frozen=True):
    """Complete declared inputs for one reference LIF experiment."""

    dataset_id: str = Field(min_length=1)
    snapshot_path: Path
    seed: int
    steps: int = Field(gt=0)
    dt_ms: float = Field(gt=0)
    tau_ms: float = Field(gt=0)
    rest_mv: float
    reset_mv: float
    threshold_mv: float
    synaptic_scale: float = Field(gt=0)
    stimuli: tuple[Stimulus, ...]
    lesion_cell_types: tuple[str, ...] = ()


class SpikeEvent(BaseModel, frozen=True):
    """Observable emitted neurons for one condition and simulation step."""

    condition: str
    step: int
    neuron_ids: tuple[int, ...]


class ExperimentResult(BaseModel, frozen=True):
    """Metrics and records required to reproduce and inspect a run."""

    config_hash: str
    configuration: dict[str, object]
    dataset_id: str
    source_manifest_sha256: str
    seed: int
    motor_spikes: int
    lesioned_motor_spikes: int | None
    lesion_effect: float | None
    runtime_seconds: float
    peak_memory_bytes: int
    software_revision: str
    events: tuple[SpikeEvent, ...]


def _software_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _config_document(config: ExperimentConfig) -> dict[str, object]:
    return cast(dict[str, object], json.loads(config.model_dump_json()))


def _config_hash(config: ExperimentConfig) -> str:
    canonical = json.dumps(_config_document(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _currents(config: ExperimentConfig, graph: SparseConnectome) -> list[np.ndarray]:
    rows = [np.zeros(graph.neuron_count, dtype=np.float32) for _ in range(config.steps)]
    index = {int(neuron_id): position for position, neuron_id in enumerate(graph.neuron_ids)}
    for stimulus in config.stimuli:
        if stimulus.step >= config.steps:
            raise ValueError(f"stimulus step {stimulus.step} exceeds experiment duration")
        try:
            position = index[stimulus.neuron_id]
        except KeyError as error:
            raise ValueError(f"stimulus targets unknown neuron {stimulus.neuron_id}") from error
        rows[stimulus.step][position] += np.float32(stimulus.current)
    return rows


def _run_condition(
    condition: str,
    config: ExperimentConfig,
    graph: SparseConnectome,
    silenced: np.ndarray | None,
) -> tuple[int, tuple[SpikeEvent, ...]]:
    params = LIFParameters(
        dt_ms=config.dt_ms,
        tau_ms=config.tau_ms,
        rest_mv=config.rest_mv,
        reset_mv=config.reset_mv,
        threshold_mv=config.threshold_mv,
    )
    motor_ids = {
        int(neuron_id)
        for neuron_id, role in zip(graph.neuron_ids, graph.roles, strict=True)
        if role == "motor"
    }
    batches = simulate_lif(
        graph,
        params,
        _currents(config, graph),
        seed=config.seed,
        silenced=silenced,
    )
    events = tuple(
        SpikeEvent(
            condition=condition,
            step=batch.step,
            neuron_ids=tuple(int(neuron_id) for neuron_id in batch.neuron_ids),
        )
        for batch in batches
    )
    motor_spikes = sum(
        neuron_id in motor_ids for event in events for neuron_id in event.neuron_ids
    )
    return motor_spikes, events


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run normal and optional cell-type-lesioned conditions as a pair."""

    metadata = SnapshotMetadata.model_validate_json(
        (config.snapshot_path / "metadata.json").read_text()
    )
    if metadata.dataset_id != config.dataset_id:
        raise ValueError(
            f"snapshot dataset {metadata.dataset_id} does not match {config.dataset_id}"
        )

    graph = SparseConnectome.from_snapshot(config.snapshot_path)
    if config.synaptic_scale != 1.0:
        graph.adjacency.data *= np.float32(config.synaptic_scale)

    tracemalloc.start()
    started = time.perf_counter()
    normal_spikes, normal_events = _run_condition("normal", config, graph, None)
    lesioned_spikes: int | None = None
    lesioned_events: tuple[SpikeEvent, ...] = ()
    if config.lesion_cell_types:
        selected = set(config.lesion_cell_types)
        mask = silence_mask(graph, lambda neuron: neuron.cell_type in selected)
        lesioned_spikes, lesioned_events = _run_condition("lesioned", config, graph, mask)
    runtime = time.perf_counter() - started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lesion_effect = (
        float(normal_spikes - lesioned_spikes) if lesioned_spikes is not None else None
    )
    return ExperimentResult(
        config_hash=_config_hash(config),
        configuration=_config_document(config),
        dataset_id=config.dataset_id,
        source_manifest_sha256=metadata.source_manifest_sha256,
        seed=config.seed,
        motor_spikes=normal_spikes,
        lesioned_motor_spikes=lesioned_spikes,
        lesion_effect=lesion_effect,
        runtime_seconds=runtime,
        peak_memory_bytes=peak_memory,
        software_revision=_software_revision(),
        events=normal_events + lesioned_events,
    )
