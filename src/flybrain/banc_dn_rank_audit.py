"""Rank bilateral descending types by input to the locked BANC DNa02 relays."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.feather as feather

from flybrain.banc_route_audit import (
    DOI,
    EXPECTED_MD5,
    THRESHOLD,
    _digests,
    _relevant_edges,
)
from flybrain.mb_association import _software_revision

PROTOCOL = "banc-v888-bilateral-dn-relay-rank-v1"
DESIGN_REVISION = "4de4fb26e3631f0c342a7bb7e13fa459220cee66"
RELAY_TYPES = {"IN19A003", "IN08A006"}
SEGMENTS = {"T1", "T2", "T3"}
SIDES = {"left", "right"}
META_COLUMNS = ("banc_888_id", "cell_type", "side", "neuromere", "proofread", "super_class")


def _cohort(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relays: list[dict[str, Any]] = []
    for row in rows:
        if row.get("proofread") != "TRUE":
            continue
        kind = row.get("cell_type")
        if not isinstance(kind, str) or not kind:
            continue
        if row.get("super_class") == "descending":
            by_type[kind].append(row)
        if kind in RELAY_TYPES:
            relays.append(row)
    sources = [
        row
        for kind in sorted(by_type)
        if len(by_type[kind]) == 2
        and {cell.get("side") for cell in by_type[kind]} == SIDES
        for row in by_type[kind]
    ]
    expected_slots = {(kind, side, segment) for kind in RELAY_TYPES
                      for side in SIDES for segment in SEGMENTS}
    actual_slots = [
        (row.get("cell_type"), row.get("side"), row.get("neuromere"))
        for row in relays
    ]
    if len(actual_slots) != 12 or set(actual_slots) != expected_slots:
        raise ValueError("BANC relay type/side/segment inventory is not exactly 12")
    selected = [*sources, *relays]
    ids = [row.get("banc_888_id") for row in selected]
    if any(not isinstance(root, str) or not root.isdigit() for root in ids):
        raise ValueError("selected BANC IDs must be decimal strings")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected BANC root ID")
    if not any(row["cell_type"] == "DNa02" for row in sources):
        raise ValueError("DNa02 is absent from bilateral descending cohort")
    return sources, sorted(relays, key=lambda row: (
        str(row["cell_type"]), str(row["side"]), str(row["neuromere"])
    ))


def _rank(
    sources: list[dict[str, Any]], relays: list[dict[str, Any]],
    edges: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {str(row["banc_888_id"]): row for row in sources}
    relay_by_id = {str(row["banc_888_id"]): row for row in relays}
    scores: dict[str, dict[str, Any]] = {}
    for source in sources:
        kind = str(source["cell_type"])
        if kind not in scores:
            scores[kind] = {"cell_type": kind, "coverage": 0, "synapses": 0}
    observed_pairs: set[tuple[str, str]] = set()
    dna02_edges = []
    for edge in edges:
        pre, post = str(edge["pre"]), str(edge["post"])
        pair = (pre, post)
        if pair in observed_pairs:
            raise ValueError("duplicate directed BANC edge")
        observed_pairs.add(pair)
        if pre not in source_by_id or post not in relay_by_id:
            raise ValueError("edge outside selected source/relay cohort")
        count = edge["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("BANC edge count must be a positive integer")
        source, relay = source_by_id[pre], relay_by_id[post]
        if source["side"] != relay["side"] or count < THRESHOLD:
            continue
        kind = str(source["cell_type"])
        scores[kind]["coverage"] += 1
        scores[kind]["synapses"] += count
        if kind == "DNa02":
            dna02_edges.append({
                "pre": pre, "post": post, "count": count,
                "side": str(source["side"]), "relay_type": str(relay["cell_type"]),
                "segment": str(relay["neuromere"]),
            })
    ranking = sorted(scores.values(), key=lambda row: (
        -int(row["coverage"]), -int(row["synapses"]), str(row["cell_type"])
    ))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    dna02 = next(row for row in ranking if row["cell_type"] == "DNa02")
    return {
        "ranking": ranking,
        "dna02": dna02,
        "dna02_edges": sorted(dna02_edges, key=lambda row: (
            row["side"], row["segment"], row["relay_type"]
        )),
        "full_coverage_types": [row["cell_type"] for row in ranking
                                if row["coverage"] == 12],
    }


def audit(metadata: Path, edges_v2: Path, edges_v3: Path) -> dict[str, Any]:
    started = time.monotonic()
    paths = {"metadata": metadata, "v2": edges_v2, "v3": edges_v3}
    digests = {name: _digests(path) for name, path in paths.items()}
    if any(digests[name]["md5"] != EXPECTED_MD5[name] for name in paths):
        raise ValueError("BANC Dataverse v3.0 source MD5 mismatch")
    table = feather.read_table(metadata, columns=list(META_COLUMNS))
    sources, relays = _cohort(table.to_pylist())
    source_ids = {str(row["banc_888_id"]) for row in sources}
    target_ids = {str(row["banc_888_id"]) for row in relays}
    versions = {
        name: _rank(sources, relays, _relevant_edges(paths[name], source_ids, target_ids))
        for name in ("v2", "v3")
    }
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "software_revision": _software_revision(),
        "source": digests,
        "dataverse_doi": DOI,
        "dataverse_version": "3.0",
        "materialization": "BANC v888",
        "edge_threshold": THRESHOLD,
        "evidence_kind": "independent_connectome_anatomy",
        "physiology_claim_allowed": False,
        "behavioral_claim_allowed": False,
        "metadata_rows": table.num_rows,
        "eligible_type_count": len(sources) // 2,
        "source_cells": sources,
        "relay_cells": relays,
        "versions": versions,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--edges-v2", type=Path, required=True)
    parser.add_argument("--edges-v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = audit(args.metadata, args.edges_v2, args.edges_v3)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
