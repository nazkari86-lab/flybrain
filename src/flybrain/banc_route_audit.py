"""Independent, provenance-locked BANC anatomy audit for the DNa02 leg route."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from flybrain.mb_association import _software_revision

PROTOCOL = "banc-v888-dna02-rotator-muscle-anatomy-v1"
DESIGN_REVISION = "59a6b9a"
DOI = "10.7910/DVN/7WTH1N"
DATAVERSE_VERSION = "3.0"
THRESHOLD = 5
EXPECTED_MD5 = {
    "metadata": "6275eda42f98c49539d1ab513d979d09",
    "v2": "394406f8a9bdf093c895f95aff4f6c49",
    "v3": "08542b0771db7418ed474be60dc9886c",
}
RELAY_TO_MUSCLE = {
    "IN19A003": "sternal_posterior_rotator_muscle",
    "IN08A006": "sternal_anterior_rotator_muscle",
}
SELECTED_TYPES = {"DNa02", "DNg13", *RELAY_TO_MUSCLE}
META_COLUMNS = (
    "banc_888_id", "cell_type", "side", "neuromere", "root_region",
    "proofread", "roughly_proofread", "flow", "super_class",
    "peripheral_target_type", "neurotransmitter_predicted",
    "neurotransmitter_verified", "manc_cell_type", "fanc_cell_type",
)


def _digests(path: Path) -> dict[str, Any]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _selected_cells(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("proofread") != "TRUE":
            continue
        if row.get("cell_type") in SELECTED_TYPES:
            selected.append(row)
        elif (
            row.get("peripheral_target_type") in RELAY_TO_MUSCLE.values()
            and row.get("flow") == "efferent"
        ):
            selected.append(row)
    ids = [row["banc_888_id"] for row in selected]
    if any(not isinstance(value, str) or not value.isdigit() for value in ids):
        raise ValueError("BANC root IDs must be decimal strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate selected BANC root ID")
    return selected


def _segment(row: dict[str, Any], *, infer_root_region: bool) -> str | None:
    value = row.get("neuromere")
    if value in {"T1", "T2", "T3"}:
        return str(value)
    if value is not None or not infer_root_region:
        return None
    region = row.get("root_region")
    side = row.get("side")
    if not isinstance(region, str) or side not in {"left", "right"}:
        return None
    hits = re.findall(r"_(T[123])_([LR])(?:_|$)", region)
    expected_side = "L" if side == "left" else "R"
    if len(hits) != 1 or hits[0][1] != expected_side:
        return None
    return str(hits[0][0])


def _relevant_edges(
    path: Path, source_ids: set[str], target_ids: set[str]
) -> Iterator[dict[str, Any]]:
    source_values = pa.array(sorted(source_ids), type=pa.string())
    target_values = pa.array(sorted(target_ids), type=pa.string())
    with pa.memory_map(str(path), "r") as stream:
        reader = pa.ipc.open_file(stream)
        if not {"pre", "post", "count"}.issubset(reader.schema.names):
            raise ValueError("BANC edgelist schema lacks pre/post/count")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column(batch.schema.get_field_index("pre"))
            post = batch.column(batch.schema.get_field_index("post"))
            mask = pc.and_(
                pc.is_in(pre, value_set=source_values),
                pc.is_in(post, value_set=target_values),
            )
            for row in batch.filter(mask).select(["pre", "post", "count"]).to_pylist():
                if type(row["count"]) is not int or row["count"] <= 0:
                    raise ValueError("BANC edge count must be a positive integer")
                yield row


def _summary(edges: list[dict[str, Any]], *, denominator: int) -> dict[str, Any]:
    passing = [edge for edge in edges if int(edge["count"]) >= THRESHOLD]
    return {
        "eligible_pairs_or_segments": denominator,
        "observed_edges": len(edges),
        "passing_edges": len(passing),
        "passing_synapses": sum(int(edge["count"]) for edge in passing),
        "passing_post_ids": sorted({str(edge["post"]) for edge in passing}),
    }


def _classify_edges(
    cells: list[dict[str, Any]], edge_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row["banc_888_id"]): row for row in cells}
    sources = [row for row in cells if row.get("cell_type") == "DNa02"]
    comparators = [row for row in cells if row.get("cell_type") == "DNg13"]
    relays = [row for row in cells if row.get("cell_type") in RELAY_TO_MUSCLE]
    motors = [
        row for row in cells
        if row.get("peripheral_target_type") in RELAY_TO_MUSCLE.values()
        and row.get("flow") == "efferent"
    ]
    if len(sources) != 2 or len(comparators) != 2 or len(relays) != 12:
        raise ValueError("expected two DNa02, two DNg13 and twelve relay cells")
    expected_relay_slots = {
        (kind, side, segment)
        for kind in RELAY_TO_MUSCLE
        for side in ("left", "right")
        for segment in ("T1", "T2", "T3")
    }
    actual_relay_slots = {
        (str(row["cell_type"]), str(row["side"]), str(row["neuromere"]))
        for row in relays
    }
    if actual_relay_slots != expected_relay_slots:
        raise ValueError("relay side/segment inventory is incomplete")
    motor_ids = {str(row["banc_888_id"]) for row in motors}
    relay_ids = {str(row["banc_888_id"]) for row in relays}
    raw = sorted(edge_rows, key=lambda row: (str(row["pre"]), str(row["post"])))
    if len({(row["pre"], row["post"]) for row in raw}) != len(raw):
        raise ValueError("duplicate directed BANC edge")
    dna02: list[dict[str, Any]] = []
    dng13: list[dict[str, Any]] = []
    output_primary: list[dict[str, Any]] = []
    output_with_inference: list[dict[str, Any]] = []
    for edge in raw:
        pre, post = str(edge["pre"]), str(edge["post"])
        if pre not in by_id or post not in by_id:
            raise ValueError("edge endpoint outside selected BANC cells")
        source, target = by_id[pre], by_id[post]
        if source.get("side") != target.get("side"):
            continue
        if post in relay_ids:
            if source.get("cell_type") == "DNa02":
                dna02.append(edge)
            elif source.get("cell_type") == "DNg13":
                dng13.append(edge)
        elif pre in relay_ids and post in motor_ids:
            required = RELAY_TO_MUSCLE[str(source["cell_type"])]
            if target.get("peripheral_target_type") != required:
                continue
            segment = str(source["neuromere"])
            if _segment(target, infer_root_region=False) == segment:
                output_primary.append(edge)
                output_with_inference.append(edge)
            elif _segment(target, infer_root_region=True) == segment:
                output_with_inference.append(edge)
    primary_slots = {
        (str(relay["cell_type"]), str(relay["side"]), str(relay["neuromere"]))
        for relay in relays
        if any(
            motor.get("peripheral_target_type") == RELAY_TO_MUSCLE[str(relay["cell_type"])]
            and motor.get("side") == relay.get("side")
            and _segment(motor, infer_root_region=False) == relay.get("neuromere")
            for motor in motors
        )
    }
    inferred_slots = {
        (str(relay["cell_type"]), str(relay["side"]), str(relay["neuromere"]))
        for relay in relays
        if any(
            motor.get("peripheral_target_type") == RELAY_TO_MUSCLE[str(relay["cell_type"])]
            and motor.get("side") == relay.get("side")
            and _segment(motor, infer_root_region=True) == relay.get("neuromere")
            for motor in motors
        )
    }

    def output_summary(
        edges: list[dict[str, Any]], slots: set[tuple[str, str, str]]
    ) -> dict[str, Any]:
        result = _summary(edges, denominator=len(slots))
        covered = sorted({
            (
                str(by_id[str(edge["pre"])]["cell_type"]),
                str(by_id[str(edge["pre"])]["side"]),
                str(by_id[str(edge["pre"])]["neuromere"]),
            )
            for edge in edges if int(edge["count"]) >= THRESHOLD
        })
        result["eligible_slots"] = [list(slot) for slot in sorted(slots)]
        result["covered_slots"] = [list(slot) for slot in covered]
        return result

    return {
        "relevant_edges": raw,
        "dna02_to_same_side_relays": {
            **_summary(dna02, denominator=12), "edges": dna02,
        },
        "dng13_to_same_side_relays": {
            **_summary(dng13, denominator=12), "edges": dng13,
        },
        "relay_to_muscle_primary": {
            **output_summary(output_primary, primary_slots), "edges": output_primary,
        },
        "relay_to_muscle_with_root_region_inference": {
            **output_summary(output_with_inference, inferred_slots),
            "edges": output_with_inference,
        },
    }


def audit(
    metadata: Path, edges_v2: Path, edges_v3: Path
) -> dict[str, Any]:
    paths = {"metadata": metadata, "v2": edges_v2, "v3": edges_v3}
    source = {name: _digests(path) for name, path in paths.items()}
    if any(source[name]["md5"] != EXPECTED_MD5[name] for name in paths):
        raise ValueError("BANC Dataverse v3.0 source MD5 mismatch")
    table = feather.read_table(metadata, columns=list(META_COLUMNS))
    cells = _selected_cells(table.to_pylist())
    source_ids = {
        str(row["banc_888_id"]) for row in cells
        if row.get("cell_type") in SELECTED_TYPES
    }
    target_ids = {
        str(row["banc_888_id"]) for row in cells
        if row.get("cell_type") in RELAY_TO_MUSCLE
        or row.get("peripheral_target_type") in RELAY_TO_MUSCLE.values()
    }
    versions = {
        name: _classify_edges(cells, _relevant_edges(paths[name], source_ids, target_ids))
        for name in ("v2", "v3")
    }
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "software_revision": _software_revision(),
        "evidence_kind": "independent_connectome_anatomy",
        "physiology_claim_allowed": False,
        "behavioral_claim_allowed": False,
        "dataverse_doi": DOI,
        "dataverse_version": DATAVERSE_VERSION,
        "materialization": "BANC v888",
        "edge_threshold": THRESHOLD,
        "source": source,
        "metadata_rows": table.num_rows,
        "selected_cells": cells,
        "versions": versions,
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
