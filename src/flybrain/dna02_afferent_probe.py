"""Reproducible afferent-isolated DNa02 thoracic motor-edge assay."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.dng33_edge_probe import _condition, _graph_digest
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import ShiuParameters, poisson_voltage_events

SOURCE_IDS = (10360, 523769)
SNAPSHOT_SHA256 = "9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f"
FROZEN_JSON_SHA256 = "3d2dc99233197e45d2849aa6b52220689e2b230fcb53f85bce991ac1ba8b0dc5"
PROTOCOL_LOCK_REVISION = "e0f2457a24d2ff51fdeebd1c71a3c482e8e90a91"


def load_frozen_edges(path: Path, *, expected_sha256: str = FROZEN_JSON_SHA256) -> dict[str, Any]:
    """Verify bytes of the previously published edge selection before use."""

    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen edge artifact SHA-256 mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("frozen edge artifact must contain an object")
    return payload


def _resolved_positions(
    graph: EventConnectome,
    frozen: dict[str, Any],
    by_id: dict[int, int],
    *,
    name: str,
    post_ids: tuple[int, ...],
    expected_superclass: str,
) -> np.ndarray:
    edges = frozen.get(name)
    if not isinstance(edges, list) or len(edges) != 17:
        raise ValueError(f"frozen edge selection must contain 17 {name}")
    positions: list[int] = []
    seen_posts: set[int] = set()
    for edge in edges:
        try:
            source = int(edge["source_id"])
            post = int(edge["post_id"])
            position = int(edge["position"])
            weight = float(edge["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frozen edge record is malformed") from exc
        if source not in SOURCE_IDS or post not in post_ids or post in seen_posts:
            raise ValueError("frozen edge identities are invalid or duplicated")
        if post not in by_id or graph.superclasses[by_id[post]] != expected_superclass:
            raise ValueError("frozen edge post cell has the wrong superclass")
        start = int(graph.outgoing.indptr[by_id[source]])
        stop = int(graph.outgoing.indptr[by_id[source] + 1])
        if (
            not start <= position < stop
            or int(graph.outgoing.indices[position]) != by_id[post]
            or float(graph.outgoing.data[position]) != weight
            or weight <= 0.0
        ):
            raise ValueError("frozen edge differs from the canonical graph")
        positions.append(position)
        seen_posts.add(post)
    if len(set(positions)) != 17 or seen_posts != set(post_ids):
        raise ValueError("frozen edge selection does not cover its declared posts")
    return np.asarray(sorted(positions), dtype=np.int64)


def run_dna02_afferent_probe(
    graph: EventConnectome,
    frozen: dict[str, Any],
    *,
    steps: int = 2_000,
    seeds: tuple[int, ...] = (31, 32, 33),
    expected_afferent_count: int | None = None,
    expected_target_weight: int | None = None,
    expected_control_weight: int | None = None,
) -> dict[str, Any]:
    """Run five paired conditions without mutating the canonical graph."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if expected_afferent_count is not None and expected_afferent_count < 0:
        raise ValueError("expected afferent count must be non-negative")
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    if len(by_id) != graph.neuron_count:
        raise ValueError("graph neuron IDs are not unique")
    try:
        frozen_sources = tuple(frozen.get("source_ids", ()))
    except TypeError as exc:
        raise ValueError("frozen DNa02 source IDs are malformed") from exc
    if frozen_sources != SOURCE_IDS:
        raise ValueError("frozen DNa02 source IDs differ")
    if any(
        source not in by_id or graph.superclasses[by_id[source]] != "descending_neuron"
        for source in SOURCE_IDS
    ):
        raise ValueError("DNa02 sources must be annotated descending neurons")
    try:
        targets = tuple(int(value) for value in frozen.get("target_ids", ()))
        controls = tuple(int(value) for value in frozen.get("control_ids", ()))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen target and control IDs are malformed") from exc
    if (
        len(targets) != 17 or len(controls) != 17
        or len(set(targets)) != 17 or len(set(controls)) != 17
        or set(targets) & set(controls)
    ):
        raise ValueError("frozen target and control IDs must be disjoint sets of 17")
    before = _graph_digest(graph)
    if frozen.get("graph_digest") != before:
        raise ValueError("frozen graph digest differs from the canonical graph")
    target_positions = _resolved_positions(
        graph, frozen, by_id, name="target_edges", post_ids=targets,
        expected_superclass="vnc_motor",
    )
    control_positions = _resolved_positions(
        graph, frozen, by_id, name="control_edges", post_ids=controls,
        expected_superclass="vnc_intrinsic",
    )
    target_weight = sum(float(edge["weight"]) for edge in frozen["target_edges"])
    control_weight = sum(float(edge["weight"]) for edge in frozen["control_edges"])
    if expected_target_weight is not None and target_weight != expected_target_weight:
        raise ValueError("frozen target edge weight sum changed")
    if expected_control_weight is not None and control_weight != expected_control_weight:
        raise ValueError("frozen control edge weight sum changed")
    source_indices = np.asarray([by_id[value] for value in SOURCE_IDS], dtype=np.int64)
    afferent = np.flatnonzero(np.isin(graph.outgoing.indices, source_indices)).astype(
        np.int64
    )
    if expected_afferent_count is not None and afferent.size != expected_afferent_count:
        raise ValueError("source afferent edge count differs from the locked anatomy")
    if set(afferent) & (set(target_positions) | set(control_positions)):
        raise ValueError("source afferent edges overlap target or control edges")
    if set(target_positions) & set(control_positions):
        raise ValueError("target and control edges overlap")
    positions = {
        "intact": afferent,
        "direct_edges_zero": np.union1d(afferent, target_positions),
        "matched_edges_zero": np.union1d(afferent, control_positions),
        "no_source": afferent,
        "replay": afferent,
    }
    incoming_counts = {
        str(source): int(np.sum(graph.outgoing.indices == by_id[source]))
        for source in SOURCE_IDS
    }
    other_motor = {
        int(graph.neuron_ids[i])
        for i, superclass in enumerate(graph.superclasses)
        if superclass == "vnc_motor" and int(graph.neuron_ids[i]) not in targets
    }
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        events = poisson_voltage_events(
            source_indices, steps=steps, params=parameters, seed=seed
        )
        conditions = {
            name: _condition(
                graph,
                name=name,
                source_ids=SOURCE_IDS,
                target_ids=targets,
                other_motor_ids=other_motor,
                events=events if name != "no_source" else {},
                edge_positions=positions[name],
                steps=steps,
                seed=seed,
                parameters=parameters,
            )
            for name in (
                "intact", "direct_edges_zero", "matched_edges_zero",
                "no_source", "replay",
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
        gates = {
            "source_and_target_recruited": intact.source_spikes > 0
            and intact_total > sum(baseline.target_counts.values()),
            "paired_source_events": intact.source_event_digest
            == direct.source_event_digest == matched.source_event_digest,
            "paired_source_spikes": intact.source_spike_digest
            == direct.source_spike_digest == matched.source_spike_digest,
            "direct_reduction_at_least_20pct": direct_reduction is not None
            and direct_reduction >= 0.2,
            "selective_over_matched_at_least_10pp": direct_reduction is not None
            and matched_reduction is not None
            and direct_reduction - matched_reduction >= 0.1,
            "replay_exact": intact.model_dump(exclude={"name"})
            == conditions["replay"].model_dump(exclude={"name"}),
            "graph_unchanged": _graph_digest(graph) == before,
        }
        rows.append({
            "seed": seed,
            "conditions": {
                name: condition.model_dump(mode="json")
                for name, condition in conditions.items()
            },
            "direct_reduction_fraction": direct_reduction,
            "matched_reduction_fraction": matched_reduction,
            "gates": gates,
            "primary_gate_passed": all(gates.values()),
        })
    return {
        "protocol": "dna02-afferent-isolated-fixed-source-direct-thoracic-edge-v1",
        "evidence_kind": "isolated_neural_simulation",
        "autonomous_behavior_claim_allowed": False,
        "graph_neurons": graph.neuron_count,
        "graph_edges": graph.edge_count,
        "graph_digest": before,
        "graph_unchanged": _graph_digest(graph) == before,
        "steps": steps,
        "shiu_parameters": dataclasses.asdict(parameters),
        "source_ids": SOURCE_IDS,
        "target_ids": targets,
        "control_ids": controls,
        "source_afferent_zero_positions": afferent.tolist(),
        "source_incoming_counts": incoming_counts,
        "target_edges": frozen["target_edges"],
        "control_edges": frozen["control_edges"],
        "seeds": rows,
        "primary_gate_passed": all(row["primary_gate_passed"] for row in rows),
    }


def main(argv: list[str] | None = None) -> int:
    """Publish a non-overwriting retained-snapshot result with locked inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--frozen-edge-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    args = parser.parse_args(argv)
    if not args.snapshot.is_dir():
        parser.error("snapshot must be an existing directory")
    if not args.frozen_edge_artifact.is_file():
        parser.error("frozen edge artifact must be an existing file")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite it")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique non-negative integers")
    snapshot_sha256 = snapshot_content_sha256(args.snapshot)
    if snapshot_sha256 != SNAPSHOT_SHA256:
        parser.error("snapshot content SHA-256 differs from the locked MaleCNS graph")
    try:
        frozen = load_frozen_edges(args.frozen_edge_artifact)
        graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(args.snapshot))
        if graph.neuron_count != 166_606 or graph.edge_count != 6_240_402:
            raise ValueError("graph dimensions differ from the locked MaleCNS snapshot")
        payload = run_dna02_afferent_probe(
            graph, frozen, steps=args.steps, seeds=tuple(args.seeds),
            expected_afferent_count=1_136,
            expected_target_weight=874,
            expected_control_weight=887,
        )
        if payload["source_incoming_counts"] != {"10360": 579, "523769": 557}:
            raise ValueError("source-specific afferent counts differ from locked anatomy")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    payload["snapshot_content_sha256"] = snapshot_sha256
    payload["frozen_edge_source_json_sha256"] = FROZEN_JSON_SHA256
    payload["protocol_lock_revision"] = PROTOCOL_LOCK_REVISION
    payload["software_revision"] = _software_revision()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
