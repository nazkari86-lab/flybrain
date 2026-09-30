"""Prospective ORN-driven control of the DNa02-linked inhibitory premotor output."""

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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

from flybrain.dna02_afferent_probe import SNAPSHOT_SHA256, SOURCE_IDS
from flybrain.dna02_odor_probe import ODOR_JSON_SHA256, load_odor_sources
from flybrain.dna02_same_post_probe import _condition
from flybrain.dng33_edge_probe import _event_digest, _graph_digest
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import ShiuParameters, poisson_voltage_events, simulate_shiu

PROTOCOL = "dna02-premotor-output-same-post-v1"
DESIGN_REVISION = "f60c836"
PRIOR_SEEDS = (40, 41, 42)
HOLDOUT_SEEDS = (43, 44, 45)
ODORS = ("food", "threat")
RELAY_TYPES = {"IN19A003", "IN08A006"}
MOTOR_BY_RELAY = {
    "IN19A003": "Sternal posterior rotator MN",
    "IN08A006": "Sternal anterior rotator MN",
}


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _edge_record(
    graph: EventConnectome, by_id: dict[int, int], pre: int, post: int
) -> dict[str, Any]:
    if pre not in by_id or post not in by_id:
        raise ValueError("declared edge endpoint is absent from graph")
    start = int(graph.outgoing.indptr[by_id[pre]])
    stop = int(graph.outgoing.indptr[by_id[pre] + 1])
    hits = np.flatnonzero(graph.outgoing.indices[start:stop] == by_id[post])
    if hits.size != 1:
        raise ValueError("declared edge must exist exactly once")
    position = start + int(hits[0])
    return {
        "source_id": pre,
        "post_id": post,
        "position": position,
        "weight": float(graph.outgoing.data[position]),
    }


def _route_and_candidates(
    graph: EventConnectome, snapshot: Path
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], tuple[int, ...], tuple[int, ...]]:
    columns = ["bodyId", "type", "superclass", "somaSide", "somaNeuromere", "statusLabel"]
    annotations = ds.dataset(snapshot / "source-annotations.parquet").to_table(
        columns=columns
    ).to_pylist()
    by_annotation = {int(row["bodyId"]): row for row in annotations}
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    if len(by_id) != graph.neuron_count:
        raise ValueError("graph neuron IDs are not unique")
    relays = tuple(sorted(
        cell_id for cell_id, row in by_annotation.items()
        if row["type"] in RELAY_TYPES
    ))
    motors = {
        cell_id for cell_id, row in by_annotation.items()
        if row["type"] in MOTOR_BY_RELAY.values()
    }
    if len(relays) != 12 or len(motors) != 34 or any(
        by_annotation[cell_id]["statusLabel"] != "Reviewed"
        or graph.transmitters[by_id[cell_id]] != "gaba"
        for cell_id in relays
    ):
        raise ValueError("published relay/motor annotation identity differs")
    if any(
        by_annotation[cell_id]["type"] != "DNa02"
        or graph.transmitters[by_id[cell_id]] != "acetylcholine"
        for cell_id in SOURCE_IDS
    ):
        raise ValueError("published DNa02 source identity differs")
    for source in SOURCE_IDS:
        for relay in relays:
            if by_annotation[source]["somaSide"] == by_annotation[relay]["somaSide"]:
                if _edge_record(graph, by_id, source, relay)["weight"] <= 0:
                    raise ValueError("DNa02-to-relay edge is not positive")
    route_rows = ds.dataset(snapshot / "edges.parquet").to_table(
        columns=["pre_id", "post_id", "sign"],
        filter=ds.field("pre_id").isin(list(relays)),
    ).to_pylist()
    route: list[dict[str, Any]] = []
    for row in route_rows:
        pre, post = int(row["pre_id"]), int(row["post_id"])
        if post not in motors:
            continue
        source_annotation = by_annotation[pre]
        post_annotation = by_annotation[post]
        if post_annotation["type"] != MOTOR_BY_RELAY[source_annotation["type"]]:
            continue
        if (
            row["sign"] != -1
            or source_annotation["somaSide"] != post_annotation["somaSide"]
            or source_annotation["somaNeuromere"] != post_annotation["somaNeuromere"]
            or post_annotation["statusLabel"] != "Reviewed"
        ):
            raise ValueError("published inhibitory route anatomy differs")
        record = _edge_record(graph, by_id, pre, post)
        if record["weight"] >= 0:
            raise ValueError("published relay output is not inhibitory-signed")
        route.append(record)
    route.sort(key=lambda row: (row["post_id"], row["source_id"]))
    targets = tuple(sorted({int(row["post_id"]) for row in route}))
    if len(route) != 33 or len(targets) != 33 or {
        int(row["source_id"]) for row in route
    } != set(relays):
        raise ValueError("published 33-edge route differs")
    incoming = ds.dataset(snapshot / "edges.parquet").to_table(
        columns=["pre_id", "post_id", "sign"],
        filter=ds.field("post_id").isin(list(targets)),
    ).to_pylist()
    candidates: dict[int, list[dict[str, Any]]] = {post: [] for post in targets}
    for row in incoming:
        pre, post = int(row["pre_id"]), int(row["post_id"])
        annotation = by_annotation.get(pre)
        if (
            pre in relays or annotation is None or row["sign"] != -1
            or annotation["superclass"] != "vnc_intrinsic"
            or graph.transmitters[by_id[pre]] != "gaba"
        ):
            continue
        record = _edge_record(graph, by_id, pre, post)
        if record["weight"] < 0:
            candidates[post].append(record)
    for post, values in candidates.items():
        values.sort(key=lambda row: row["source_id"])
        if len(values) < 2:
            raise ValueError(f"not enough same-post GABA candidates for {post}")
    return route, candidates, relays, targets


def _source_indices(
    graph: EventConnectome, odor_artifact: dict[str, Any]
) -> dict[str, np.ndarray]:
    if (
        odor_artifact.get("steps") != 2_000
        or set(odor_artifact.get("source_ids", {})) != set(ODORS)
    ):
        raise ValueError("registered-ORN artifact is malformed")
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    result: dict[str, np.ndarray] = {}
    for odor in ODORS:
        values = tuple(int(value) for value in odor_artifact["source_ids"][odor])
        if not values or len(values) != len(set(values)) or any(
            value not in by_id or not graph.cell_types[by_id[value]].startswith("ORN_")
            for value in values
        ):
            raise ValueError("registered-ORN source identity differs")
        result[odor] = np.asarray([by_id[value] for value in sorted(values)], dtype=np.int64)
    if set(odor_artifact["source_ids"]["food"]) & set(odor_artifact["source_ids"]["threat"]):
        raise ValueError("registered-ORN populations overlap")
    return result


def _control_score(
    edges: tuple[dict[str, Any], ...],
    *,
    vectors: dict[int, tuple[float, ...]],
    target_vector: tuple[float, ...],
    target_weight: float,
) -> tuple[float, float, tuple[int, ...]]:
    ids = tuple(sorted(int(edge["source_id"]) for edge in edges))
    exposure = tuple(sum(vectors[source][i] for source in ids) for i in range(6))
    return (
        sum(abs(exposure[i] - target_vector[i]) for i in range(6)),
        abs(sum(abs(float(edge["weight"])) for edge in edges) - target_weight),
        ids,
    )


def _select_controls(
    route: list[dict[str, Any]], candidates: dict[int, list[dict[str, Any]]],
    panel_counts: list[Counter[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(panel_counts) != 6:
        raise ValueError("control selection requires six prior intact panels")
    singles: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    for target in route:
        post = int(target["post_id"])
        target_weight = abs(float(target["weight"]))
        target_vector = tuple(
            target_weight * counts[int(target["source_id"])] for counts in panel_counts
        )
        values = candidates[post]

        def vector(edge: dict[str, Any]) -> tuple[float, ...]:
            return tuple(
                abs(float(edge["weight"])) * counts[int(edge["source_id"])]
                for counts in panel_counts
            )

        vectors = {int(edge["source_id"]): vector(edge) for edge in values}

        score = partial(
            _control_score, vectors=vectors, target_vector=target_vector,
            target_weight=target_weight,
        )
        single = min(values, key=lambda edge: score((edge,)))
        pair = min(itertools.combinations(values, 2), key=score)
        singles.append(single)
        pairs.extend(pair)
        exposures.append({
            "post_id": post,
            "route": target_vector,
            "single": vector(single),
            "pair": tuple(sum(vector(edge)[i] for edge in pair) for i in range(6)),
        })
    return singles, pairs, exposures


def freeze_controls(
    graph: EventConnectome, snapshot: Path, odor_artifact: dict[str, Any]
) -> dict[str, Any]:
    route, candidates, relays, targets = _route_and_candidates(graph, snapshot)
    sources = _source_indices(graph, odor_artifact)
    watch = set(relays) | {
        int(edge["source_id"]) for values in candidates.values() for edge in values
    }
    counts: list[Counter[int]] = []
    event_digests: list[dict[str, Any]] = []
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    for odor in ODORS:
        for seed in PRIOR_SEEDS:
            events = poisson_voltage_events(
                sources[odor], steps=2_000, params=parameters, seed=seed
            )
            observed: Counter[int] = Counter()
            for batch in simulate_shiu(
                graph, parameters, steps=2_000, external_voltage_events=events, seed=seed
            ):
                observed.update(int(value) for value in batch.neuron_ids if int(value) in watch)
            counts.append(observed)
            event_digests.append({"odor": odor, "seed": seed, "digest": _event_digest(events)})
    singles, pairs, exposure = _select_controls(route, candidates, counts)
    selection = {"route": route, "single": singles, "pair": pairs, "prior_exposure": exposure}
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "selection_digest": _canonical_digest(selection),
        "selection": selection,
        "prior_event_digests": event_digests,
        "prior_seeds": PRIOR_SEEDS,
        "holdout_seeds": HOLDOUT_SEEDS,
        "relay_ids": relays,
        "target_ids": targets,
        "graph_digest": _graph_digest(graph),
        "snapshot_content_sha256": SNAPSHOT_SHA256,
        "odor_artifact_json_sha256": ODOR_JSON_SHA256,
        "software_revision": _software_revision(),
    }


def _validated_control_positions(
    graph: EventConnectome, snapshot: Path, frozen: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    tuple[int, ...], tuple[int, ...],
]:
    route, candidates, relays, targets = _route_and_candidates(graph, snapshot)
    selection = frozen.get("selection")
    if (
        frozen.get("protocol") != PROTOCOL
        or frozen.get("design_revision") != DESIGN_REVISION
        or frozen.get("snapshot_content_sha256") != SNAPSHOT_SHA256
        or frozen.get("odor_artifact_json_sha256") != ODOR_JSON_SHA256
        or frozen.get("graph_digest") != _graph_digest(graph)
        or frozen.get("selection_digest") != _canonical_digest(selection)
        or frozen.get("prior_seeds") != list(PRIOR_SEEDS)
        or frozen.get("holdout_seeds") != list(HOLDOUT_SEEDS)
        or frozen.get("relay_ids") != list(relays)
        or frozen.get("target_ids") != list(targets)
        or not isinstance(selection, dict)
        or selection.get("route") != route
    ):
        raise ValueError("frozen control identity differs from published protocol")
    singles = selection.get("single")
    pairs = selection.get("pair")
    if (
        not isinstance(singles, list) or not isinstance(pairs, list)
        or len(singles) != 33 or len(pairs) != 66
    ):
        raise ValueError("frozen same-post control counts differ")
    for name, values, count in (("single", singles, 1), ("pair", pairs, 2)):
        grouped: dict[int, list[int]] = {post: [] for post in targets}
        for edge in values:
            if not isinstance(edge, dict) or edge.get("post_id") not in grouped:
                raise ValueError(f"{name} control target differs")
            post = int(edge["post_id"])
            if edge not in candidates[post]:
                raise ValueError(f"{name} control edge differs from canonical graph")
            grouped[post].append(int(edge["source_id"]))
        if any(len(ids) != count or len(ids) != len(set(ids)) for ids in grouped.values()):
            raise ValueError(f"{name} control per-target cardinality differs")
    if set(int(edge["position"]) for edge in route) & set(
        int(edge["position"]) for edge in singles + pairs
    ):
        raise ValueError("route and control edges overlap")
    return route, singles, pairs, relays, targets


def run_holdout(
    graph: EventConnectome, snapshot: Path, odor_artifact: dict[str, Any],
    frozen: dict[str, Any], frozen_sha256: str,
) -> dict[str, Any]:
    route, singles, pairs, relays, targets = _validated_control_positions(
        graph, snapshot, frozen
    )
    sources = _source_indices(graph, odor_artifact)
    control_sources = tuple(sorted(set(relays) | {
        int(edge["source_id"]) for edge in singles + pairs
    }))
    other_motor = {
        int(graph.neuron_ids[i]) for i, superclass in enumerate(graph.superclasses)
        if superclass == "vnc_motor" and int(graph.neuron_ids[i]) not in targets
    }
    positions = {
        "route_zero": np.asarray(sorted(int(x["position"]) for x in route), dtype=np.int64),
        "single_zero": np.asarray(sorted(int(x["position"]) for x in singles), dtype=np.int64),
        "pair_zero": np.asarray(sorted(int(x["position"]) for x in pairs), dtype=np.int64),
    }
    if len(set(positions["pair_zero"])) != 66:
        raise ValueError("pair-control CSR positions are duplicated")
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    before = _graph_digest(graph)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for odor in ODORS:
        for seed in HOLDOUT_SEEDS:
            events = poisson_voltage_events(
                sources[odor], steps=2_000, params=parameters, seed=seed
            )
            conditions = {
                name: _condition(
                    graph, name=name, events={} if name == "no_source" else events,
                    positions=positions.get(name), targets=targets,
                    control_sources=control_sources, other_motor=other_motor,
                    steps=2_000, seed=seed, parameters=parameters,
                )
                for name in (
                    "intact", "route_zero", "single_zero", "pair_zero", "no_source", "replay"
                )
            }
            intact = conditions["intact"]
            quiet = conditions["no_source"]
            paired = [
                conditions[name]
                for name in ("intact", "route_zero", "single_zero", "pair_zero")
            ]
            base = sum(intact["target_counts"].values())
            increases = {
                name: (sum(conditions[name]["target_counts"].values()) - base) / base
                if base else None
                for name in ("route_zero", "single_zero", "pair_zero")
            }
            route_exposure = sum(
                abs(float(edge["weight"]))
                * intact["control_source_counts"][int(edge["source_id"])]
                for edge in route
            )
            pair_exposure = sum(
                abs(float(edge["weight"]))
                * intact["control_source_counts"][int(edge["source_id"])]
                for edge in pairs
            )
            ratio = pair_exposure / route_exposure if route_exposure else None
            route_increase = increases["route_zero"]
            single_increase = increases["single_zero"]
            pair_increase = increases["pair_zero"]
            gates = {
                "paired_events": len({x["source_event_digest"] for x in paired}) == 1,
                "source_recruited": intact["dna02_spikes"] > quiet["dna02_spikes"]
                and all(intact["control_source_counts"][relay] > 0 for relay in relays)
                and base > sum(quiet["target_counts"].values()),
                "no_source_quiet": quiet["dna02_spikes"] == 0
                and all(quiet["control_source_counts"][relay] == 0 for relay in relays)
                and sum(quiet["target_counts"].values()) == 0,
                "presynaptic_timing_paired": len({x["dna02_spike_digest"] for x in paired}) == 1
                and len({x["control_source_spike_digest"] for x in paired}) == 1,
                "other_motor_equal": len({x["other_motor_spikes"] for x in paired}) == 1,
                "route_increase_at_least_25pct": route_increase is not None
                and route_increase >= 0.25,
                "route_margin_over_both_at_least_10pp": route_increase is not None
                and single_increase is not None and pair_increase is not None
                and route_increase - max(single_increase, pair_increase) >= 0.10,
                "pair_exposure_80_to_120pct": ratio is not None and 0.8 <= ratio <= 1.2,
                "replay_exact": {k: v for k, v in intact.items() if k != "name"}
                == {k: v for k, v in conditions["replay"].items() if k != "name"},
                "graph_unchanged": _graph_digest(graph) == before,
            }
            rows.append({
                "odor": odor, "seed": seed, "conditions": conditions,
                "increases": increases,
                "pair_to_route_exposure_ratio": ratio,
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
        "shiu_parameters": dataclasses.asdict(parameters),
        "odor_source_ids": odor_artifact["source_ids"],
        "dna02_ids": SOURCE_IDS,
        "relay_ids": relays,
        "target_ids": targets,
        "route_edges": route,
        "single_control_edges": singles,
        "pair_control_edges": pairs,
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
            controls = json.loads(raw)
            result = run_holdout(
                graph, args.snapshot, odor, controls, hashlib.sha256(raw).hexdigest()
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
