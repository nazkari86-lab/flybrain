"""Fixed-source, sparse-edge DNg33 motor-target localization in MaleCNS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import ShiuParameters, VoltageEvents, poisson_voltage_events, simulate_shiu

SOURCE_IDS = (13317, 13442)
TARGET_IDS = (800659, 803732, 810086, 813291, 814430, 814989, 815205, 815281)
CONTROL_IDS = (802110, 802174, 803352, 805489, 802652, 802945, 904983, 805244)


class FrozenEdge(BaseModel, frozen=True):
    """One dataset edge selected without changing the canonical CSR."""

    model_config = ConfigDict(extra="forbid")

    source_id: int
    post_id: int
    position: int
    weight: float = Field(gt=0)


class EdgeCondition(BaseModel, frozen=True):
    """One neural trajectory with a specific edge overlay."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_voltage_events: int = Field(ge=0)
    source_event_digest: str
    source_spikes: int = Field(ge=0)
    source_spike_digest: str
    target_counts: dict[int, int]
    other_motor_spikes: int = Field(ge=0)
    total_spikes: int = Field(ge=0)
    trace_digest: str


class EdgeSeedResult(BaseModel, frozen=True):
    """Five source-paired conditions and the prespecified per-seed gates."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    conditions: dict[str, EdgeCondition]
    direct_reduction_fraction: float | None
    matched_reduction_fraction: float | None
    gates: dict[str, bool]
    primary_gate_passed: bool


class Dng33EdgeProbeResult(BaseModel, frozen=True):
    """Fail-closed model-level evidence; never a behavioral intelligence claim."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["dng33-fixed-source-direct-edge-v1"] = (
        "dng33-fixed-source-direct-edge-v1"
    )
    evidence_kind: Literal["isolated_neural_simulation"] = "isolated_neural_simulation"
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    graph_digest: str
    graph_unchanged: bool
    steps: int = Field(gt=0)
    shiu_parameters: dict[str, float]
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    control_ids: tuple[int, ...]
    target_edges: tuple[FrozenEdge, ...]
    control_edges: tuple[FrozenEdge, ...]
    seeds: tuple[EdgeSeedResult, ...]
    primary_gate_passed: bool
    autonomous_behavior_claim_allowed: Literal[False] = False


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for values in (
        graph.neuron_ids,
        graph.outgoing.indptr,
        graph.outgoing.indices,
        graph.outgoing.data,
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def resolve_frozen_edges(
    graph: EventConnectome,
    source_ids: tuple[int, ...],
    post_ids: tuple[int, ...],
) -> tuple[FrozenEdge, ...]:
    """Resolve biological IDs to exact positive CSR positions, failing closed."""

    if (
        not source_ids
        or not post_ids
        or len(set(source_ids)) != len(source_ids)
        or len(set(post_ids)) != len(post_ids)
    ):
        raise ValueError("source and post IDs must be nonempty and unique")
    by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    if len(by_id) != graph.neuron_count:
        raise ValueError("graph neuron IDs are not unique")
    edges: list[FrozenEdge] = []
    for source_id in source_ids:
        source_index = by_id.get(source_id)
        if source_index is None:
            raise ValueError(f"source ID absent from graph: {source_id}")
        start = int(graph.outgoing.indptr[source_index])
        stop = int(graph.outgoing.indptr[source_index + 1])
        for post_id in post_ids:
            post_index = by_id.get(post_id)
            if post_index is None:
                raise ValueError(f"post ID absent from graph: {post_id}")
            matches = np.flatnonzero(graph.outgoing.indices[start:stop] == post_index)
            if matches.size != 1:
                raise ValueError(
                    f"declared edge must exist exactly once and be positive: {source_id}→{post_id}"
                )
            position = start + int(matches[0])
            weight = float(graph.outgoing.data[position])
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"declared edge must be positive: {source_id}→{post_id}")
            edges.append(
                FrozenEdge(source_id=source_id, post_id=post_id, position=position, weight=weight)
            )
    return tuple(edges)


def _event_digest(events: VoltageEvents) -> str:
    digest = hashlib.sha256()
    for step, (indices, voltages) in sorted(events.items()):
        digest.update(np.asarray([step, len(indices)], dtype="<i8").tobytes())
        digest.update(indices.astype("<i8", copy=False).tobytes())
        digest.update(voltages.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _condition(
    graph: EventConnectome,
    *,
    name: str,
    source_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    other_motor_ids: set[int],
    events: VoltageEvents,
    edge_positions: np.ndarray | None,
    steps: int,
    seed: int,
    parameters: ShiuParameters,
) -> EdgeCondition:
    source_set = set(source_ids)
    target_set = set(target_ids)
    counts: Counter[int] = Counter()
    source_digest = hashlib.sha256()
    trace_digest = hashlib.sha256()
    total_spikes = 0
    overlay = (
        np.zeros(edge_positions.size, dtype=np.float32) if edge_positions is not None else None
    )
    for batch in simulate_shiu(
        graph,
        parameters,
        steps=steps,
        external_voltage_events=events,
        seed=seed,
        plastic_edge_indices=edge_positions,
        plastic_edge_multipliers=overlay,
    ):
        fired = tuple(int(value) for value in batch.neuron_ids)
        trace_digest.update(np.asarray([batch.step, len(fired)], dtype="<i8").tobytes())
        trace_digest.update(batch.neuron_ids.astype("<u8", copy=False).tobytes())
        source_fired = tuple(value for value in fired if value in source_set)
        if source_fired:
            source_digest.update(np.asarray([batch.step, len(source_fired)], dtype="<i8").tobytes())
            source_digest.update(np.asarray(source_fired, dtype="<u8").tobytes())
        counts.update(
            value
            for value in fired
            if value in source_set or value in target_set or value in other_motor_ids
        )
        total_spikes += len(fired)
    return EdgeCondition(
        name=name,
        source_voltage_events=sum(len(indices) for indices, _ in events.values()),
        source_event_digest=_event_digest(events),
        source_spikes=sum(counts[value] for value in source_ids),
        source_spike_digest=source_digest.hexdigest(),
        target_counts={value: counts[value] for value in target_ids},
        other_motor_spikes=sum(counts[value] for value in other_motor_ids),
        total_spikes=total_spikes,
        trace_digest=trace_digest.hexdigest(),
    )


def run_dng33_edge_probe(
    graph: EventConnectome,
    *,
    steps: int = 2_000,
    seeds: tuple[int, ...] = (19, 20, 21),
) -> Dng33EdgeProbeResult:
    """Run the prospectively locked five-condition panel on one immutable graph."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    for value in SOURCE_IDS:
        if value not in by_id or graph.superclasses[by_id[value]] != "descending_neuron":
            raise ValueError("DNg33 source must be an annotated descending neuron")
    for values, superclass in ((TARGET_IDS, "vnc_motor"), (CONTROL_IDS, "vnc_intrinsic")):
        if any(
            value not in by_id or graph.superclasses[by_id[value]] != superclass
            for value in values
        ):
            raise ValueError(f"declared postsynaptic cells must be {superclass}")
    target_edges = resolve_frozen_edges(graph, SOURCE_IDS, TARGET_IDS)
    control_edges = resolve_frozen_edges(graph, SOURCE_IDS, CONTROL_IDS)
    if len(target_edges) != 16 or sum(edge.weight for edge in target_edges) != 404:
        raise ValueError("declared direct-edge anatomy must retain sum 404")
    if len(control_edges) != 16 or sum(edge.weight for edge in control_edges) != 379:
        raise ValueError("declared control-edge anatomy must retain sum 379")
    if set(edge.position for edge in target_edges) & set(edge.position for edge in control_edges):
        raise ValueError("target and control edge sets overlap")
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    source_indices = np.asarray([by_id[value] for value in SOURCE_IDS], dtype=np.int64)
    other_motor_ids = {
        int(graph.neuron_ids[index])
        for index, superclass in enumerate(graph.superclasses)
        if superclass == "vnc_motor" and int(graph.neuron_ids[index]) not in TARGET_IDS
    }
    before = _graph_digest(graph)
    target_positions = np.asarray(sorted(edge.position for edge in target_edges), dtype=np.int64)
    control_positions = np.asarray(sorted(edge.position for edge in control_edges), dtype=np.int64)
    seed_results: list[EdgeSeedResult] = []
    for seed in seeds:
        events = poisson_voltage_events(source_indices, steps=steps, params=parameters, seed=seed)
        condition_args: dict[str, Any] = {
            "graph": graph,
            "source_ids": SOURCE_IDS,
            "target_ids": TARGET_IDS,
            "other_motor_ids": other_motor_ids,
            "steps": steps,
            "seed": seed,
            "parameters": parameters,
        }
        conditions = {
            name: _condition(
                **condition_args,
                name=name,
                events=events if name != "no_source" else {},
                edge_positions=positions,
            )
            for name, positions in (
                ("intact", None),
                ("direct_edges_zero", target_positions),
                ("matched_edges_zero", control_positions),
                ("no_source", None),
                ("replay", None),
            )
        }
        intact = conditions["intact"]
        direct = conditions["direct_edges_zero"]
        matched = conditions["matched_edges_zero"]
        baseline = conditions["no_source"]
        intact_total = sum(intact.target_counts.values())
        direct_total = sum(direct.target_counts.values())
        matched_total = sum(matched.target_counts.values())
        direct_reduction = (
            (intact_total - direct_total) / intact_total if intact_total else None
        )
        matched_reduction = (
            (intact_total - matched_total) / intact_total if intact_total else None
        )
        graph_unchanged = _graph_digest(graph) == before
        gates = {
            "source_and_target_recruited": intact.source_spikes > 0
            and intact_total > sum(baseline.target_counts.values()),
            "paired_source_events": intact.source_event_digest
            == direct.source_event_digest
            == matched.source_event_digest,
            "paired_source_spikes": intact.source_spike_digest
            == direct.source_spike_digest
            == matched.source_spike_digest,
            "direct_reduction_at_least_20pct": direct_reduction is not None
            and direct_reduction >= 0.2,
            "selective_over_matched_at_least_10pp": direct_reduction is not None
            and matched_reduction is not None
            and direct_reduction - matched_reduction >= 0.1,
            "replay_exact": intact.model_dump(exclude={"name"})
            == conditions["replay"].model_dump(exclude={"name"}),
            "graph_unchanged": graph_unchanged,
        }
        seed_results.append(
            EdgeSeedResult(
                seed=seed,
                conditions=conditions,
                direct_reduction_fraction=direct_reduction,
                matched_reduction_fraction=matched_reduction,
                gates=gates,
                primary_gate_passed=all(gates.values()),
            )
        )
    return Dng33EdgeProbeResult(
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        graph_digest=before,
        graph_unchanged=_graph_digest(graph) == before,
        steps=steps,
        shiu_parameters={
            "dt_ms": parameters.dt_ms,
            "refractory_ms": parameters.refractory_ms,
            "synaptic_delay_ms": parameters.synaptic_delay_ms,
            "poisson_rate_hz": parameters.poisson_rate_hz,
            "poisson_voltage_scale": parameters.poisson_voltage_scale,
            "synapse_mv": parameters.synapse_mv,
        },
        source_ids=SOURCE_IDS,
        target_ids=TARGET_IDS,
        control_ids=CONTROL_IDS,
        target_edges=target_edges,
        control_edges=control_edges,
        seeds=tuple(seed_results),
        primary_gate_passed=all(result.primary_gate_passed for result in seed_results),
    )


def main(argv: list[str] | None = None) -> int:
    """Write a non-overwriting, provenance-rich JSON for a retained snapshot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 20, 21])
    args = parser.parse_args(argv)
    if not args.snapshot.is_dir():
        parser.error("snapshot must be an existing directory")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite it")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique non-negative integers")
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(args.snapshot))
    result = run_dng33_edge_probe(graph, steps=args.steps, seeds=tuple(args.seeds))
    payload = result.model_dump(mode="json")
    payload["snapshot_content_sha256"] = snapshot_content_sha256(args.snapshot)
    payload["software_revision"] = _software_revision()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
