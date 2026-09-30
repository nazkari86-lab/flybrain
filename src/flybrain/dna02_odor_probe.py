"""Registered-ORN input to intact DNa02 direct thoracic edges."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import resource
import sys
import time
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
from flybrain.dng33_edge_probe import _condition, _graph_digest
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import _software_revision
from flybrain.odor_mbon_probe import _graph_digest as _odor_graph_digest
from flybrain.provenance import snapshot_content_sha256
from flybrain.shiu import ShiuParameters, poisson_voltage_events

ODOR_JSON_SHA256 = "3f3493acb4b61fa8cedb2f5cc8cb07c84b4367335161e2a9a7045c7c35f3864e"
PROTOCOL_LOCK_REVISION = "67720fa87c258ffaf7a790d6cc59f5999b3bb1f7"
ODORS = ("food", "threat")


def load_odor_sources(path: Path) -> dict[str, Any]:
    """Load the published registered-ORN source IDs only after byte verification."""

    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ODOR_JSON_SHA256:
        raise ValueError("registered-ORN artifact SHA-256 mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("source_ids"), dict):
        raise ValueError("registered-ORN artifact is malformed")
    return payload


def run_dna02_odor_probe(
    graph: EventConnectome,
    frozen: dict[str, Any],
    odor_artifact: dict[str, Any],
    *,
    steps: int = 2_000,
    seeds: tuple[int, ...] = (37, 38, 39),
) -> dict[str, Any]:
    """Run fixed-source paired edge interventions without changing the graph."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    before = _graph_digest(graph)
    if (
        frozen.get("graph_digest") != before
        or odor_artifact.get("graph_digest") != _odor_graph_digest(graph)
    ):
        raise ValueError("source artifact graph digest differs from the canonical graph")
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    if len(by_id) != graph.neuron_count:
        raise ValueError("graph neuron IDs are not unique")
    if tuple(frozen.get("source_ids", ())) != SOURCE_IDS:
        raise ValueError("frozen DNa02 sources differ")
    try:
        targets = tuple(int(value) for value in frozen["target_ids"])
        controls = tuple(int(value) for value in frozen["control_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen edge identities are malformed") from exc
    if (
        len(targets) != 17 or len(controls) != 17
        or len(set(targets)) != 17 or len(set(controls)) != 17
        or set(targets) & set(controls)
    ):
        raise ValueError("frozen target and control sets must be disjoint sets of 17")
    direct = _resolved_positions(
        graph, frozen, by_id, name="target_edges", post_ids=targets,
        expected_superclass="vnc_motor",
    )
    matched = _resolved_positions(
        graph, frozen, by_id, name="control_edges", post_ids=controls,
        expected_superclass="vnc_intrinsic",
    )
    if set(direct) & set(matched):
        raise ValueError("direct and control edges overlap")
    if odor_artifact.get("steps") != 2_000:
        raise ValueError("registered-ORN source artifact has changed steps")
    source_lists = odor_artifact.get("source_ids")
    if not isinstance(source_lists, dict) or set(source_lists) != set(ODORS):
        raise ValueError("registered-ORN source populations are malformed")
    source_indices: dict[str, np.ndarray] = {}
    source_sets: dict[str, set[int]] = {}
    for odor in ODORS:
        try:
            source_ids = tuple(int(value) for value in source_lists[odor])
        except (TypeError, ValueError) as exc:
            raise ValueError("registered-ORN source IDs are malformed") from exc
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("registered-ORN source IDs must be nonempty and unique")
        if any(
            value not in by_id or not graph.cell_types[by_id[value]].startswith("ORN_")
            for value in source_ids
        ):
            raise ValueError("registered-ORN source is absent or not annotated ORN")
        source_sets[odor] = set(source_ids)
        source_indices[odor] = np.asarray(
            [by_id[value] for value in sorted(source_ids)], dtype=np.int64
        )
    if source_sets["food"] & source_sets["threat"]:
        raise ValueError("registered-ORN source populations overlap")
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
                    graph, name=name, source_ids=SOURCE_IDS, target_ids=targets,
                    other_motor_ids=other_motor, events={} if name == "no_source" else events,
                    edge_positions=positions, steps=steps, seed=seed, parameters=parameters,
                )
                for name, positions in (
                    ("intact", None),
                    ("direct_edges_zero", direct),
                    ("matched_edges_zero", matched),
                    ("no_source", None),
                    ("replay", None),
                )
            }
            intact = conditions["intact"]
            direct_run = conditions["direct_edges_zero"]
            matched_run = conditions["matched_edges_zero"]
            quiet = conditions["no_source"]
            target_total = sum(intact.target_counts.values())
            direct_total = sum(direct_run.target_counts.values())
            matched_total = sum(matched_run.target_counts.values())
            direct_reduction = (
                (target_total - direct_total) / target_total if target_total else None
            )
            matched_reduction = (
                (target_total - matched_total) / target_total if target_total else None
            )
            gates = {
                "paired_orn_events": intact.source_event_digest
                == direct_run.source_event_digest == matched_run.source_event_digest,
                "dna02_and_targets_recruited": intact.source_spikes > quiet.source_spikes
                and target_total > sum(quiet.target_counts.values()),
                "direct_pair_source_timing_equal": intact.source_spike_digest
                == direct_run.source_spike_digest,
                "direct_pair_other_motor_equal": intact.other_motor_spikes
                == direct_run.other_motor_spikes,
                "direct_reduction_at_least_5pct": direct_reduction is not None
                and direct_reduction >= 0.05,
                "no_source_quiet": quiet.source_spikes == 0
                and sum(quiet.target_counts.values()) == 0,
                "replay_exact": intact.model_dump(exclude={"name"})
                == conditions["replay"].model_dump(exclude={"name"}),
                "graph_unchanged": _graph_digest(graph) == before,
            }
            rows.append({
                "odor": odor,
                "seed": seed,
                "conditions": {
                    name: condition.model_dump(mode="json")
                    for name, condition in conditions.items()
                },
                "direct_reduction_fraction": direct_reduction,
                "matched_reduction_fraction": matched_reduction,
                "matched_source_timing_equal": intact.source_spike_digest
                == matched_run.source_spike_digest,
                "gates": gates,
                "narrow_gate_passed": all(gates.values()),
            })
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "protocol": "dna02-registered-orn-direct-edge-v1",
        "evidence_kind": "isolated_neural_simulation",
        "autonomous_behavior_claim_allowed": False,
        "graph_neurons": graph.neuron_count,
        "graph_edges": graph.edge_count,
        "graph_digest": before,
        "registered_orn_graph_digest": _odor_graph_digest(graph),
        "graph_unchanged": _graph_digest(graph) == before,
        "steps": steps,
        "shiu_parameters": dataclasses.asdict(parameters),
        "source_ids": SOURCE_IDS,
        "odor_source_ids": {odor: sorted(source_sets[odor]) for odor in ODORS},
        "target_ids": targets,
        "control_ids": controls,
        "target_edges": frozen["target_edges"],
        "control_edges": frozen["control_edges"],
        "seeds": rows,
        "narrow_gate_passed": all(row["narrow_gate_passed"] for row in rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }


def main(argv: list[str] | None = None) -> int:
    """Write an immutable JSON result from the locked retained inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--frozen-edge-artifact", type=Path, required=True)
    parser.add_argument("--odor-source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[37, 38, 39])
    args = parser.parse_args(argv)
    if not args.snapshot.is_dir():
        parser.error("snapshot must be an existing directory")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite it")
    if args.steps != 2_000 or args.seeds != [37, 38, 39]:
        parser.error("published protocol locks 2,000 steps and seeds 37 38 39")
    if snapshot_content_sha256(args.snapshot) != SNAPSHOT_SHA256:
        parser.error("snapshot SHA-256 differs from the locked MaleCNS graph")
    try:
        frozen = load_frozen_edges(args.frozen_edge_artifact)
        odor_artifact = load_odor_sources(args.odor_source_artifact)
        graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(args.snapshot))
        if graph.neuron_count != 166_606 or graph.edge_count != 6_240_402:
            raise ValueError("graph dimensions differ from the locked MaleCNS snapshot")
        result = run_dna02_odor_probe(graph, frozen, odor_artifact)
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
