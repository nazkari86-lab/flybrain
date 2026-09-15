"""Measured whole-connectome smoke experiments for Shiu reference dynamics."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.shiu import ShiuParameters, poisson_voltage_events, simulate_shiu


class ShiuSmokeMetrics(BaseModel, frozen=True):
    """Auditable activity and resource measurements for one smoke run."""

    snapshot: str
    neurons: int
    edges: int
    unresolved_sign_edges: int
    stimulated_sensory_neurons: int
    external_events: int
    duration_ms: float
    steps: int
    seed: int
    total_spikes: int
    reached_neurons: int
    spikes_by_role: dict[str, int]
    parameters: dict[str, float]
    runtime_seconds: float
    peak_rss_bytes: int
    event_graph_storage_bytes: int
    activity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _unresolved_edge_count(snapshot: Path) -> int:
    metadata = json.loads((snapshot / "metadata.json").read_text())
    return int(metadata.get("unresolved_sign_edges", 0))


def run_shiu_smoke(
    snapshot: Path,
    *,
    duration_ms: float,
    seed: int,
    params: ShiuParameters | None = None,
) -> ShiuSmokeMetrics:
    """Stimulate all annotated sensory neurons with the published Poisson input."""

    import hashlib

    active_params = params or ShiuParameters()
    active_params.validate()
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    steps_float = duration_ms / active_params.dt_ms
    steps = round(steps_float)
    if not np.isclose(steps_float, steps):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")

    started = time.perf_counter()
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    sensory_indices = np.fromiter(
        (index for index, role in enumerate(graph.roles) if role == "sensory"),
        dtype=np.int64,
    )
    events = poisson_voltage_events(
        sensory_indices,
        steps=steps,
        params=active_params,
        seed=seed,
    )
    refractory_exempt = np.zeros(graph.neuron_count, dtype=np.bool_)
    refractory_exempt[sensory_indices] = True

    role_by_id = {
        int(neuron_id): role
        for neuron_id, role in zip(graph.neuron_ids, graph.roles, strict=True)
    }
    reached: set[int] = set()
    spikes_by_role: dict[str, int] = {}
    total_spikes = 0
    digest = hashlib.sha256()
    for batch in simulate_shiu(
        graph,
        active_params,
        steps=steps,
        external_voltage_events=events,
        refractory_exempt=refractory_exempt,
        seed=seed,
    ):
        digest.update(batch.step.to_bytes(8, byteorder="little", signed=False))
        digest.update(batch.neuron_ids.tobytes())
        total_spikes += int(batch.neuron_ids.size)
        for neuron_id_raw in batch.neuron_ids:
            neuron_id = int(neuron_id_raw)
            reached.add(neuron_id)
            role = role_by_id[neuron_id]
            spikes_by_role[role] = spikes_by_role.get(role, 0) + 1

    return ShiuSmokeMetrics(
        snapshot=str(snapshot.resolve()),
        neurons=graph.neuron_count,
        edges=graph.edge_count,
        unresolved_sign_edges=_unresolved_edge_count(snapshot),
        stimulated_sensory_neurons=int(sensory_indices.size),
        external_events=sum(indices.size for indices, _ in events.values()),
        duration_ms=duration_ms,
        steps=steps,
        seed=seed,
        total_spikes=total_spikes,
        reached_neurons=len(reached),
        spikes_by_role=dict(sorted(spikes_by_role.items())),
        parameters={key: float(value) for key, value in asdict(active_params).items()},
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        event_graph_storage_bytes=graph.storage_bytes,
        activity_digest=digest.hexdigest(),
    )
