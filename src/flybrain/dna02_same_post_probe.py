"""Intact ORN→DNa02 assay with frozen same-postsynaptic VNC controls."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.dna02_afferent_probe import (
    FROZEN_JSON_SHA256,
    SNAPSHOT_SHA256,
    SOURCE_IDS,
    _resolved_positions,
    load_frozen_edges,
)
from flybrain.dna02_odor_probe import ODOR_JSON_SHA256, load_odor_sources
from flybrain.dng33_edge_probe import _event_digest, _graph_digest
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.odor_mbon_probe import _graph_digest as _odor_graph_digest
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import (
    ShiuParameters,
    VoltageEvents,
    poisson_voltage_events,
    simulate_shiu,
)

PROTOCOL_LOCK_REVISION = "becb0eccc9522848c5a365e60073c634b631f65e"
ODORS = ("food", "threat")

# Published in the protocol before seed 40-42 outcomes. Keys are motor post IDs.
SINGLE_SOURCES: dict[int, int] = {
    800158: 802450,
    800316: 809067,
    801469: 906407,
    801813: 800373,
    801918: 906407,
    801946: 904800,
    906235: 800846,
    801548: 800383,
    802215: 800631,
    804851: 801111,
    806923: 803035,
    815344: 804217,
    818399: 801149,
    832490: 903653,
    903689: 802780,
    919509: 803035,
    927808: 800747,
}
PAIR_SOURCES: dict[int, tuple[int, int]] = {
    800158: (802450, 811021),
    800316: (807487, 809067),
    801469: (811006, 906407),
    801813: (800373, 810066),
    801918: (800214, 906407),
    801946: (801421, 904800),
    906235: (800846, 807487),
    801548: (800383, 801550),
    802215: (907028, 909475),
    804851: (801111, 806584),
    806923: (803035, 804647),
    815344: (800173, 904395),
    818399: (801149, 802757),
    832490: (804773, 903653),
    903689: (802780, 905387),
    919509: (800075, 803035),
    927808: (800174, 803210),
}


def _same_post_edges(
    graph: EventConnectome,
    by_id: dict[int, int],
    mapping: Mapping[int, tuple[int, ...]],
    targets: tuple[int, ...],
    *,
    count_per_target: int,
) -> tuple[list[dict[str, int | float]], np.ndarray]:
    if set(mapping) != set(targets):
        raise ValueError("same-post control targets differ from frozen targets")
    records: list[dict[str, int | float]] = []
    positions: list[int] = []
    for post in targets:
        sources = mapping[post]
        if len(sources) != count_per_target or len(set(sources)) != count_per_target:
            raise ValueError("same-post control sources are malformed")
        if post not in by_id or graph.superclasses[by_id[post]] != "vnc_motor":
            raise ValueError("same-post control target is not a motor cell")
        for source in sources:
            if source in SOURCE_IDS or source not in by_id:
                raise ValueError("same-post control source is absent or is DNa02")
            source_index = by_id[source]
            if graph.superclasses[source_index] != "vnc_intrinsic":
                raise ValueError("same-post control source is not VNC intrinsic")
            start = int(graph.outgoing.indptr[source_index])
            stop = int(graph.outgoing.indptr[source_index + 1])
            matches = np.flatnonzero(graph.outgoing.indices[start:stop] == by_id[post])
            if matches.size != 1:
                raise ValueError("same-post control edge must exist exactly once")
            position = start + int(matches[0])
            weight = float(graph.outgoing.data[position])
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("same-post control edge must have positive finite weight")
            positions.append(position)
            records.append({
                "source_id": source, "post_id": post,
                "position": position, "weight": weight,
            })
    if len(set(positions)) != len(positions):
        raise ValueError("same-post control edge positions are duplicated")
    return records, np.asarray(sorted(positions), dtype=np.int64)


def _update_spike_digest(digest: Any, step: int, ids: tuple[int, ...]) -> None:
    if ids:
        digest.update(np.asarray([step, len(ids)], dtype="<i8").tobytes())
        digest.update(np.asarray(ids, dtype="<u8").tobytes())


def _condition(
    graph: EventConnectome,
    *,
    name: str,
    events: VoltageEvents,
    positions: np.ndarray | None,
    targets: tuple[int, ...],
    control_sources: tuple[int, ...],
    other_motor: set[int],
    steps: int,
    seed: int,
    parameters: ShiuParameters,
) -> dict[str, Any]:
    dna02_set = set(SOURCE_IDS)
    control_set = set(control_sources)
    watch = dna02_set | control_set | set(targets) | other_motor
    counts: Counter[int] = Counter()
    dna02_digest = hashlib.sha256()
    control_digest = hashlib.sha256()
    trace_digest = hashlib.sha256()
    total_spikes = 0
    multipliers = np.zeros(positions.size, dtype=np.float32) if positions is not None else None
    for batch in simulate_shiu(
        graph, parameters, steps=steps, external_voltage_events=events, seed=seed,
        plastic_edge_indices=positions, plastic_edge_multipliers=multipliers,
    ):
        fired = tuple(int(value) for value in batch.neuron_ids)
        trace_digest.update(np.asarray([batch.step, len(fired)], dtype="<i8").tobytes())
        trace_digest.update(batch.neuron_ids.astype("<u8", copy=False).tobytes())
        _update_spike_digest(
            dna02_digest, batch.step, tuple(value for value in fired if value in dna02_set)
        )
        _update_spike_digest(
            control_digest, batch.step, tuple(value for value in fired if value in control_set)
        )
        counts.update(value for value in fired if value in watch)
        total_spikes += len(fired)
    return {
        "name": name,
        "source_voltage_events": sum(len(indices) for indices, _ in events.values()),
        "source_event_digest": _event_digest(events),
        "dna02_source_counts": {value: counts[value] for value in SOURCE_IDS},
        "dna02_spikes": sum(counts[value] for value in SOURCE_IDS),
        "dna02_spike_digest": dna02_digest.hexdigest(),
        "control_source_counts": {value: counts[value] for value in control_sources},
        "control_source_spike_digest": control_digest.hexdigest(),
        "target_counts": {value: counts[value] for value in targets},
        "other_motor_spikes": sum(counts[value] for value in other_motor),
        "total_spikes": total_spikes,
        "trace_digest": trace_digest.hexdigest(),
    }


def run_same_post_probe(
    graph: EventConnectome,
    frozen: dict[str, Any],
    odor_artifact: dict[str, Any],
    *,
    steps: int = 2_000,
    seeds: tuple[int, ...] = (40, 41, 42),
) -> dict[str, Any]:
    """Run paired, locked same-post lesions and retain every failed gate."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    graph_digest = _graph_digest(graph)
    if (
        frozen.get("graph_digest") != graph_digest
        or odor_artifact.get("graph_digest") != _odor_graph_digest(graph)
    ):
        raise ValueError("source artifact graph digest differs from the canonical graph")
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    if len(by_id) != graph.neuron_count or tuple(frozen.get("source_ids", ())) != SOURCE_IDS:
        raise ValueError("canonical neuron IDs or frozen DNa02 source IDs differ")
    try:
        targets = tuple(int(value) for value in frozen["target_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen motor target IDs are malformed") from exc
    if len(targets) != 17 or len(set(targets)) != 17:
        raise ValueError("frozen motor target IDs must be 17 distinct cells")
    direct_positions = _resolved_positions(
        graph, frozen, by_id, name="target_edges", post_ids=targets,
        expected_superclass="vnc_motor",
    )
    single_mapping = {post: (source,) for post, source in SINGLE_SOURCES.items()}
    single_edges, single_positions = _same_post_edges(
        graph, by_id, single_mapping, targets, count_per_target=1
    )
    pair_edges, pair_positions = _same_post_edges(
        graph, by_id, PAIR_SOURCES, targets, count_per_target=2
    )
    if set(direct_positions) & (set(single_positions) | set(pair_positions)):
        raise ValueError("direct and same-post control edges overlap")
    source_lists = odor_artifact.get("source_ids")
    if odor_artifact.get("steps") != 2_000 or not isinstance(source_lists, dict):
        raise ValueError("registered-ORN artifact is malformed")
    if set(source_lists) != set(ODORS):
        raise ValueError("registered-ORN populations differ")
    source_indices: dict[str, np.ndarray] = {}
    source_sets: dict[str, set[int]] = {}
    for odor in ODORS:
        try:
            ids = tuple(int(value) for value in source_lists[odor])
        except (TypeError, ValueError) as exc:
            raise ValueError("registered-ORN IDs are malformed") from exc
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("registered-ORN IDs must be nonempty and unique")
        if any(
            value not in by_id or not graph.cell_types[by_id[value]].startswith("ORN_")
            for value in ids
        ):
            raise ValueError("registered-ORN source is absent or not annotated ORN")
        source_sets[odor] = set(ids)
        source_indices[odor] = np.asarray([by_id[value] for value in sorted(ids)], dtype=np.int64)
    if source_sets["food"] & source_sets["threat"]:
        raise ValueError("registered-ORN source populations overlap")
    control_sources = tuple(sorted({int(edge["source_id"]) for edge in single_edges + pair_edges}))
    other_motor = {
        int(graph.neuron_ids[i])
        for i, superclass in enumerate(graph.superclasses)
        if superclass == "vnc_motor" and int(graph.neuron_ids[i]) not in targets
    }
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for odor in ODORS:
        for seed in seeds:
            events = poisson_voltage_events(
                source_indices[odor], steps=steps, params=parameters, seed=seed
            )
            conditions = {
                name: _condition(
                    graph, name=name, events={} if name == "no_source" else events,
                    positions=positions, targets=targets, control_sources=control_sources,
                    other_motor=other_motor, steps=steps, seed=seed, parameters=parameters,
                )
                for name, positions in (
                    ("intact", None),
                    ("direct_edges_zero", direct_positions),
                    ("same_post_single_zero", single_positions),
                    ("same_post_pair_zero", pair_positions),
                    ("no_source", None),
                    ("replay", None),
                )
            }
            intact = conditions["intact"]
            direct = conditions["direct_edges_zero"]
            single = conditions["same_post_single_zero"]
            pair = conditions["same_post_pair_zero"]
            quiet = conditions["no_source"]
            intact_total = sum(intact["target_counts"].values())
            reductions = {
                name: (intact_total - sum(conditions[name]["target_counts"].values()))
                / intact_total if intact_total else None
                for name in ("direct_edges_zero", "same_post_single_zero", "same_post_pair_zero")
            }
            direct_exposure = sum(
                float(edge["weight"]) * intact["dna02_source_counts"][int(edge["source_id"])]
                for edge in frozen["target_edges"]
            )
            single_exposure = sum(
                float(edge["weight"]) * intact["control_source_counts"][int(edge["source_id"])]
                for edge in single_edges
            )
            pair_exposure = sum(
                float(edge["weight"]) * intact["control_source_counts"][int(edge["source_id"])]
                for edge in pair_edges
            )
            pair_ratio = pair_exposure / direct_exposure if direct_exposure else None
            direct_reduction = reductions["direct_edges_zero"]
            single_reduction = reductions["same_post_single_zero"]
            pair_reduction = reductions["same_post_pair_zero"]
            gates = {
                "paired_events": len({
                    condition["source_event_digest"]
                    for condition in (intact, direct, single, pair)
                }) == 1,
                "dna02_and_targets_recruited": intact["dna02_spikes"] > quiet["dna02_spikes"]
                and intact_total > sum(quiet["target_counts"].values()),
                "no_source_quiet": quiet["dna02_spikes"] == 0
                and sum(quiet["target_counts"].values()) == 0,
                "source_timing_paired": len({
                    condition["dna02_spike_digest"]
                    for condition in (intact, direct, single, pair)
                }) == 1 and len({
                    condition["control_source_spike_digest"]
                    for condition in (intact, direct, single, pair)
                }) == 1,
                "other_motor_equal_all_conditions": len({
                    condition["other_motor_spikes"]
                    for condition in (intact, direct, single, pair)
                }) == 1,
                "direct_reduction_at_least_5pct": direct_reduction is not None
                and direct_reduction >= 0.05,
                "direct_margin_over_both_at_least_5pp": direct_reduction is not None
                and single_reduction is not None and pair_reduction is not None
                and direct_reduction - max(single_reduction, pair_reduction) >= 0.05,
                "pair_exposure_80_to_120pct": pair_ratio is not None
                and 0.8 <= pair_ratio <= 1.2,
                "replay_exact": {k: v for k, v in intact.items() if k != "name"}
                == {k: v for k, v in conditions["replay"].items() if k != "name"},
                "graph_unchanged": _graph_digest(graph) == graph_digest,
            }
            rows.append({
                "odor": odor,
                "seed": seed,
                "conditions": conditions,
                "reductions": reductions,
                "intact_weighted_exposure": {
                    "dna02_direct": direct_exposure,
                    "same_post_single": single_exposure,
                    "same_post_pair": pair_exposure,
                    "pair_to_direct_ratio": pair_ratio,
                },
                "gates": gates,
                "strict_gate_passed": all(gates.values()),
            })
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "protocol": "dna02-registered-orn-same-post-control-v1",
        "evidence_kind": "isolated_neural_simulation",
        "autonomous_behavior_claim_allowed": False,
        "graph_neurons": graph.neuron_count,
        "graph_edges": graph.edge_count,
        "graph_digest": graph_digest,
        "registered_orn_graph_digest": _odor_graph_digest(graph),
        "graph_unchanged": _graph_digest(graph) == graph_digest,
        "steps": steps,
        "shiu_parameters": dataclasses.asdict(parameters),
        "dna02_ids": SOURCE_IDS,
        "odor_source_ids": {odor: sorted(source_sets[odor]) for odor in ODORS},
        "target_ids": targets,
        "dna02_direct_edges": frozen["target_edges"],
        "single_control_edges": single_edges,
        "pair_control_edges": pair_edges,
        "control_source_ids": control_sources,
        "seeds": rows,
        "strict_gate_passed": all(row["strict_gate_passed"] for row in rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--frozen-edge-artifact", type=Path, required=True)
    parser.add_argument("--odor-source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    args = parser.parse_args(argv)
    if not args.snapshot.is_dir():
        parser.error("snapshot must be an existing directory")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite it")
    if args.steps != 2_000 or args.seeds != [40, 41, 42]:
        parser.error("published protocol locks 2,000 steps and seeds 40 41 42")
    if snapshot_content_sha256(args.snapshot) != SNAPSHOT_SHA256:
        parser.error("snapshot SHA-256 differs from the locked MaleCNS graph")
    try:
        frozen = load_frozen_edges(args.frozen_edge_artifact)
        odor = load_odor_sources(args.odor_source_artifact)
        graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(args.snapshot))
        if graph.neuron_count != 166_606 or graph.edge_count != 6_240_402:
            raise ValueError("graph dimensions differ from the locked MaleCNS snapshot")
        result = run_same_post_probe(graph, frozen, odor)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    result["snapshot_content_sha256"] = SNAPSHOT_SHA256
    result["frozen_edge_json_sha256"] = FROZEN_JSON_SHA256
    result["registered_orn_json_sha256"] = ODOR_JSON_SHA256
    result["protocol_lock_revision"] = PROTOCOL_LOCK_REVISION
    result["software_revision"] = _software_revision()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
