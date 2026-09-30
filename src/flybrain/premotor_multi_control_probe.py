"""Prospective activity-matched same-post control of premotor inhibition."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import itertools
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flybrain.dna02_afferent_probe import SNAPSHOT_SHA256
from flybrain.dna02_odor_probe import ODOR_JSON_SHA256, load_odor_sources
from flybrain.dna02_same_post_probe import _condition
from flybrain.dng33_edge_probe import _event_digest, _graph_digest
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.premotor_output_probe import (
    ODORS,
    PRIOR_SEEDS,
    _canonical_digest,
    _route_and_candidates,
    _source_indices,
)
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import ShiuParameters, poisson_voltage_events, simulate_shiu

PROTOCOL = "dna02-premotor-multi-input-control-v1"
DESIGN_REVISION = "67fd712"
HOLDOUT_SEEDS = (46, 47, 48)
MAX_CONTROL_EDGES_PER_POST = 4
CHUNK_SIZE = 50_000


def _chunks(values: Iterator[tuple[int, ...]]) -> Iterator[list[tuple[int, ...]]]:
    while batch := list(itertools.islice(values, CHUNK_SIZE)):
        yield batch


def _select_for_post(
    target: dict[str, Any], candidates: list[dict[str, Any]],
    panel_counts: list[Counter[int]],
) -> tuple[list[dict[str, Any]], tuple[float, ...], tuple[float, ...]]:
    """Find the exact best one-to-four edge subset by vectorized enumeration."""
    if len(panel_counts) != 6 or not candidates:
        raise ValueError("selection requires six prior panels and control candidates")
    values = sorted(candidates, key=lambda edge: int(edge["source_id"]))
    source_ids = tuple(int(edge["source_id"]) for edge in values)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate same-post control source")
    weights = np.asarray([abs(float(edge["weight"])) for edge in values], dtype=np.float64)
    vectors = np.asarray([
        [weight * counts[source_id] for counts in panel_counts]
        for weight, source_id in zip(weights, source_ids, strict=True)
    ], dtype=np.float64)
    target_weight = abs(float(target["weight"]))
    target_vector = np.asarray([
        target_weight * counts[int(target["source_id"])] for counts in panel_counts
    ], dtype=np.float64)
    best_key: tuple[float, float, int, tuple[int, ...]] | None = None
    best_indices: tuple[int, ...] = ()
    for size in range(1, min(MAX_CONTROL_EDGES_PER_POST, len(values)) + 1):
        for batch in _chunks(itertools.combinations(range(len(values)), size)):
            indices = np.asarray(batch, dtype=np.int64)
            errors = np.abs(vectors[indices].sum(axis=1) - target_vector).sum(axis=1)
            weight_errors = np.abs(weights[indices].sum(axis=1) - target_weight)
            local = int(np.lexsort((np.arange(len(batch)), weight_errors, errors))[0])
            choice = tuple(int(index) for index in indices[local])
            key = (
                float(errors[local]), float(weight_errors[local]), size,
                tuple(source_ids[index] for index in choice),
            )
            if best_key is None or key < best_key:
                best_key, best_indices = key, choice
    if best_key is None:
        raise ValueError("no eligible control subset")
    chosen = [values[index] for index in best_indices]
    control_vector = vectors[np.asarray(best_indices, dtype=np.int64)].sum(axis=0)
    return chosen, tuple(float(x) for x in target_vector), tuple(float(x) for x in control_vector)


def _prior_counts(
    graph: EventConnectome, odor_artifact: dict[str, Any], watch: set[int]
) -> tuple[list[Counter[int]], list[dict[str, Any]]]:
    sources = _source_indices(graph, odor_artifact)
    params = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    counts: list[Counter[int]] = []
    event_digests: list[dict[str, Any]] = []
    for odor in ODORS:
        for seed in PRIOR_SEEDS:
            events = poisson_voltage_events(sources[odor], steps=2_000, params=params, seed=seed)
            observed: Counter[int] = Counter()
            for batch in simulate_shiu(
                graph, params, steps=2_000, external_voltage_events=events, seed=seed
            ):
                observed.update(int(value) for value in batch.neuron_ids if int(value) in watch)
            counts.append(observed)
            event_digests.append({"odor": odor, "seed": seed, "digest": _event_digest(events)})
    return counts, event_digests


def freeze_controls(
    graph: EventConnectome, snapshot: Path, odor_artifact: dict[str, Any]
) -> dict[str, Any]:
    route, candidates, relays, targets = _route_and_candidates(graph, snapshot)
    watch = set(relays) | {
        int(edge["source_id"]) for values in candidates.values() for edge in values
    }
    started = time.perf_counter()
    counts, event_digests = _prior_counts(graph, odor_artifact, watch)
    selected: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    for target in route:
        choice, route_vector, control_vector = _select_for_post(
            target, candidates[int(target["post_id"])], counts
        )
        selected.extend(choice)
        exposures.append({
            "post_id": int(target["post_id"]),
            "route": route_vector,
            "control": control_vector,
        })
    ratios = []
    for index in range(6):
        route_total = sum(float(item["route"][index]) for item in exposures)
        control_total = sum(float(item["control"][index]) for item in exposures)
        ratios.append(control_total / route_total if route_total > 0 else None)
    selection = {"route": route, "control": selected, "prior_exposure": exposures}
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "selection_digest": _canonical_digest(selection),
        "selection": selection,
        "prior_panel_event_digests": event_digests,
        "prior_panel_exposure_ratios": ratios,
        "prior_balance_passed": all(ratio is not None and 0.8 <= ratio <= 1.2 for ratio in ratios),
        "prior_seeds": PRIOR_SEEDS,
        "holdout_seeds": HOLDOUT_SEEDS,
        "relay_ids": relays,
        "target_ids": targets,
        "graph_digest": _graph_digest(graph),
        "snapshot_content_sha256": SNAPSHOT_SHA256,
        "odor_artifact_json_sha256": ODOR_JSON_SHA256,
        "software_revision": _software_revision(),
        "runtime_seconds": time.perf_counter() - started,
    }


def _validated_selection(
    graph: EventConnectome, snapshot: Path, frozen: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, ...], tuple[int, ...]]:
    route, candidates, relays, targets = _route_and_candidates(graph, snapshot)
    selection = frozen.get("selection")
    if (
        frozen.get("protocol") != PROTOCOL
        or frozen.get("design_revision") != DESIGN_REVISION
        or frozen.get("snapshot_content_sha256") != SNAPSHOT_SHA256
        or frozen.get("odor_artifact_json_sha256") != ODOR_JSON_SHA256
        or frozen.get("graph_digest") != _graph_digest(graph)
        or not isinstance(selection, dict)
        or frozen.get("selection_digest") != _canonical_digest(selection)
        or selection.get("route") != route
        or frozen.get("prior_seeds") != list(PRIOR_SEEDS)
        or frozen.get("holdout_seeds") != list(HOLDOUT_SEEDS)
        or frozen.get("relay_ids") != list(relays)
        or frozen.get("target_ids") != list(targets)
    ):
        raise ValueError("frozen control identity differs from locked design")
    control = selection.get("control")
    exposure = selection.get("prior_exposure")
    if not isinstance(control, list) or not isinstance(exposure, list) or len(exposure) != 33:
        raise ValueError("frozen control structure differs")
    grouped: dict[int, list[dict[str, Any]]] = {post: [] for post in targets}
    for edge in control:
        if not isinstance(edge, dict) or edge.get("post_id") not in grouped:
            raise ValueError("control post differs")
        post = int(edge["post_id"])
        if edge not in candidates[post]:
            raise ValueError("control edge differs from canonical graph")
        grouped[post].append(edge)
    if any(not 1 <= len(values) <= 4 for values in grouped.values()):
        raise ValueError("control cardinality differs")
    positions = [int(edge["position"]) for edge in control]
    if len(set(positions)) != len(positions) or set(positions) & {
        int(edge["position"]) for edge in route
    }:
        raise ValueError("control positions duplicated or overlap route")
    expected_ratios: list[float] = []
    for index in range(6):
        route_total = sum(float(item["route"][index]) for item in exposure)
        control_total = sum(float(item["control"][index]) for item in exposure)
        if route_total <= 0:
            raise ValueError("invalid prior route exposure")
        expected_ratios.append(control_total / route_total)
    if frozen.get("prior_panel_exposure_ratios") != expected_ratios or not all(
        0.8 <= ratio <= 1.2 for ratio in expected_ratios
    ) or frozen.get("prior_balance_passed") is not True:
        raise ValueError("prior activity matching failed or was altered")
    return route, control, relays, targets


def run_holdout(
    graph: EventConnectome, snapshot: Path, odor_artifact: dict[str, Any],
    frozen: dict[str, Any], frozen_sha256: str,
) -> dict[str, Any]:
    route, control, relays, targets = _validated_selection(graph, snapshot, frozen)
    sources = _source_indices(graph, odor_artifact)
    control_sources = tuple(sorted(set(relays) | {
        int(edge["source_id"]) for edge in control
    }))
    other_motor = {
        int(graph.neuron_ids[i]) for i, superclass in enumerate(graph.superclasses)
        if superclass == "vnc_motor" and int(graph.neuron_ids[i]) not in targets
    }
    route_positions: NDArray[np.int64] = np.asarray(
        sorted(int(edge["position"]) for edge in route), dtype=np.int64
    )
    control_positions: NDArray[np.int64] = np.asarray(
        sorted(int(edge["position"]) for edge in control), dtype=np.int64
    )
    params = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    before = _graph_digest(graph)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for odor in ODORS:
        for seed in HOLDOUT_SEEDS:
            events = poisson_voltage_events(sources[odor], steps=2_000, params=params, seed=seed)
            positions_by_name = {
                "route_zero": route_positions,
                "control_zero": control_positions,
            }
            conditions = {
                name: _condition(
                    graph, name=name, events={} if name == "no_source" else events,
                    positions=positions_by_name.get(name),
                    targets=targets, control_sources=control_sources, other_motor=other_motor,
                    steps=2_000, seed=seed, parameters=params,
                )
                for name in ("intact", "route_zero", "control_zero", "no_source", "replay")
            }
            intact, route_zero, control_zero = (
                conditions[name] for name in ("intact", "route_zero", "control_zero")
            )
            quiet = conditions["no_source"]
            paired = [intact, route_zero, control_zero]
            base = sum(intact["target_counts"].values())
            route_increase = (
                (sum(route_zero["target_counts"].values()) - base) / base if base else None
            )
            control_increase = (
                (sum(control_zero["target_counts"].values()) - base) / base if base else None
            )
            route_exposure = sum(
                abs(float(edge["weight"]))
                * intact["control_source_counts"][int(edge["source_id"])]
                for edge in route
            )
            control_exposure = sum(
                abs(float(edge["weight"]))
                * intact["control_source_counts"][int(edge["source_id"])]
                for edge in control
            )
            ratio = control_exposure / route_exposure if route_exposure else None
            gates = {
                "paired_events": len({item["source_event_digest"] for item in paired}) == 1,
                "source_recruited": intact["dna02_spikes"] > quiet["dna02_spikes"]
                and all(intact["control_source_counts"][relay] > 0 for relay in relays)
                and base > sum(quiet["target_counts"].values()),
                "no_source_quiet": quiet["dna02_spikes"] == 0
                and all(quiet["control_source_counts"][relay] == 0 for relay in relays)
                and sum(quiet["target_counts"].values()) == 0,
                "presynaptic_timing_paired": len({
                    item["dna02_spike_digest"] for item in paired
                }) == 1
                and len({item["control_source_spike_digest"] for item in paired}) == 1,
                "other_motor_equal": len({item["other_motor_spikes"] for item in paired}) == 1,
                "route_increase_at_least_25pct": route_increase is not None
                and route_increase >= 0.25,
                "route_margin_at_least_10pp": route_increase is not None
                and control_increase is not None
                and route_increase - control_increase >= 0.10,
                "control_exposure_80_to_120pct": ratio is not None and 0.8 <= ratio <= 1.2,
                "replay_exact": {k: v for k, v in intact.items() if k != "name"}
                == {k: v for k, v in conditions["replay"].items() if k != "name"},
                "graph_unchanged": _graph_digest(graph) == before,
            }
            rows.append({
                "odor": odor, "seed": seed, "conditions": conditions,
                "route_increase": route_increase,
                "control_increase": control_increase,
                "control_to_route_exposure_ratio": ratio,
                "gates": gates,
                "strict_gate_passed": all(gates.values()),
            })
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "protocol": PROTOCOL,
        "evidence_kind": "isolated_neural_simulation",
        "autonomous_behavior_claim_allowed": False,
        "design_revision": DESIGN_REVISION,
        "software_revision": _software_revision(),
        "snapshot_content_sha256": SNAPSHOT_SHA256,
        "odor_artifact_json_sha256": ODOR_JSON_SHA256,
        "frozen_controls_sha256": frozen_sha256,
        "frozen_selection_digest": frozen["selection_digest"],
        "graph_digest": before,
        "graph_unchanged": _graph_digest(graph) == before,
        "graph_neurons": graph.neuron_count,
        "graph_edges": graph.edge_count,
        "steps": 2_000,
        "shiu_parameters": dataclasses.asdict(params),
        "odor_source_ids": odor_artifact["source_ids"],
        "route_edges": route,
        "control_edges": control,
        "relay_ids": relays,
        "target_ids": targets,
        "seeds": rows,
        "strict_gate_passed": all(row["strict_gate_passed"] for row in rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--odor-source-artifact", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.snapshot.is_dir() or args.output.exists():
        parser.error("snapshot must exist and output must not exist")
    if args.mode == "run" and args.controls is None:
        parser.error("run mode requires --controls")
    if snapshot_content_sha256(args.snapshot) != SNAPSHOT_SHA256:
        parser.error("snapshot SHA-256 differs from locked MaleCNS")
    try:
        odor = load_odor_sources(args.odor_source_artifact)
        graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(args.snapshot))
        if graph.neuron_count != 166_606 or graph.edge_count != 6_240_402:
            raise ValueError("graph dimensions differ from locked MaleCNS")
        if args.mode == "freeze":
            result = freeze_controls(graph, args.snapshot, odor)
        else:
            assert args.controls is not None
            raw = (
                gzip.decompress(args.controls.read_bytes())
                if args.controls.suffix == ".gz" else args.controls.read_bytes()
            )
            result = run_holdout(
                graph, args.snapshot, odor, json.loads(raw), hashlib.sha256(raw).hexdigest()
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
