"""Streaming adapter for official MaleCNS flat-connectome exports."""

from __future__ import annotations

import hashlib
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from flybrain.acquire import acquire_artifact
from flybrain.manifest import load_manifest
from flybrain.schema import EDGE_SCHEMA, NEURON_SCHEMA


@dataclass(frozen=True)
class MaleCNSSources:
    """Verified local paths for the four essential MaleCNS products."""

    annotations: Path
    neurotransmitters: Path
    stats: Path
    weights: Path
    manifest_sha256: str
    dataset_id: str

    @classmethod
    def from_manifest(cls, manifest_path: Path, cache_root: Path) -> MaleCNSSources:
        manifest = load_manifest(manifest_path)
        resolved: dict[str, Path] = {}
        for artifact in manifest.artifacts:
            filename = Path(urlparse(str(artifact.url)).path).name
            if filename.startswith("body-annotations-"):
                key = "annotations"
            elif filename.startswith("body-neurotransmitters-"):
                key = "neurotransmitters"
            elif filename.startswith("body-stats-"):
                key = "stats"
            elif filename.startswith("connectome-weights-"):
                key = "weights"
            else:
                continue
            if key in resolved:
                raise ValueError(f"duplicate MaleCNS artifact role: {key}")
            resolved[key] = acquire_artifact(artifact, cache_root)

        required = {"annotations", "neurotransmitters", "stats", "weights"}
        missing = sorted(required - resolved.keys())
        if missing:
            raise ValueError(f"missing MaleCNS artifacts: {', '.join(missing)}")

        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return cls(
            annotations=resolved["annotations"],
            neurotransmitters=resolved["neurotransmitters"],
            stats=resolved["stats"],
            weights=resolved["weights"],
            manifest_sha256=manifest_sha256,
            dataset_id=manifest.dataset_id,
        )


@dataclass(frozen=True)
class MaleCNSSelectionMetrics:
    """Measured population counts from the source annotation policy."""

    source_annotation_rows: int
    selected_neurons: int
    untyped_neurons: int
    missing_transmitters: int


class MaleCNSImportMetrics(BaseModel, frozen=True):
    """Auditable source, output, biological-uncertainty, and resource counts."""

    dataset_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_annotation_rows: int
    source_weight_rows: int
    selected_neurons: int
    connected_neurons: int
    untyped_neurons: int
    missing_transmitters: int
    retained_edges: int
    retained_synapse_weight: int
    unresolved_sign_edges: int
    unresolved_sign_weight: int
    min_weight: int
    runtime_seconds: float
    peak_rss_bytes: int


def transmitter_sign(name: str) -> tuple[int, str]:
    """Return a conservative fast-sign assumption and its provenance."""

    normalized = name.strip().lower()
    if normalized == "acetylcholine":
        return 1, "fast-transmitter-assumption"
    if normalized in {"gaba", "histamine"}:
        return -1, "fast-transmitter-assumption"
    if normalized == "glutamate":
        return 0, "receptor-context-required"
    if normalized in {"dopamine", "octopamine", "serotonin"}:
        return 0, "neuromodulator-not-fast-sign"
    return 0, "unresolved-transmitter"


def _text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _role(superclass: str) -> str:
    if superclass.endswith("_sensory"):
        return "sensory"
    if superclass == "vnc_motor":
        return "motor"
    if superclass == "descending_neuron":
        return "descending"
    if superclass == "ascending_neuron":
        return "ascending"
    return "interneuron"


def _valid_annotation_mask(annotations: pa.Table) -> pa.Array:
    return pa.array(
        [
            value is not None and "tbc" not in str(value).lower()
            for value in annotations.column("superclass").to_pylist()
        ]
    )


def select_neurons(
    annotations: pa.Table,
    neurotransmitters: pa.Table,
    *,
    dataset_id: str = "male-cns-v1.0",
) -> tuple[pa.Table, MaleCNSSelectionMetrics]:
    """Apply the published valid-superclass policy and join biological metadata."""

    required_annotations = {
        "bodyId",
        "type",
        "superclass",
        "somaSide",
        "rootSide",
        "status",
        "statusLabel",
    }
    missing_annotations = sorted(required_annotations - set(annotations.column_names))
    if missing_annotations:
        raise ValueError(f"missing annotation columns: {', '.join(missing_annotations)}")
    if not {"body", "consensus_nt"}.issubset(neurotransmitters.column_names):
        raise ValueError("neurotransmitter table requires body and consensus_nt")

    selected = annotations.filter(_valid_annotation_mask(annotations))
    selected_ids = selected.column("bodyId").to_pylist()
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate selected annotation body ID")

    selected_nt = neurotransmitters.filter(
        pc.is_in(neurotransmitters.column("body"), value_set=pa.array(selected_ids))
    )
    nt_bodies = selected_nt.column("body").to_pylist()
    if len(nt_bodies) != len(set(nt_bodies)):
        raise ValueError("duplicate neurotransmitter body ID")
    nt_by_body = dict(
        zip(nt_bodies, selected_nt.column("consensus_nt").to_pylist(), strict=True)
    )

    records = selected.to_pylist()
    untyped = 0
    missing_transmitters = 0
    columns: dict[str, list[object]] = {name: [] for name in NEURON_SCHEMA.names}
    for record in records:
        body_id = int(record["bodyId"])
        cell_type = _text(record["type"], "untyped")
        if cell_type == "untyped":
            untyped += 1
        superclass = _text(record["superclass"], "unknown")
        soma_side = _text(record["somaSide"], "")
        side = soma_side or _text(record["rootSide"], "unknown")
        raw_transmitter = nt_by_body.get(body_id)
        transmitter = _text(raw_transmitter, "unclear")
        if raw_transmitter is None:
            missing_transmitters += 1
        columns["neuron_id"].append(body_id)
        columns["source_dataset"].append(dataset_id)
        columns["cell_type"].append(cell_type)
        columns["superclass"].append(superclass)
        columns["side"].append(side)
        columns["transmitter"].append(transmitter)
        columns["transmitter_provenance"].append(
            "male-cns-consensus" if raw_transmitter is not None else "missing-source-annotation"
        )
        columns["role"].append(_role(superclass))
        columns["annotation_status"].append(_text(record["status"], "unknown"))
        columns["status_label"].append(_text(record["statusLabel"], "unknown"))
        columns["annotation_confidence"].append(0.5)

    neurons = pa.Table.from_pydict(columns, schema=NEURON_SCHEMA).sort_by("neuron_id")
    metrics = MaleCNSSelectionMetrics(
        source_annotation_rows=annotations.num_rows,
        selected_neurons=neurons.num_rows,
        untyped_neurons=untyped,
        missing_transmitters=missing_transmitters,
    )
    return neurons, metrics


def _membership_positions(
    sorted_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    in_bounds = positions < sorted_ids.size
    safe_positions = np.minimum(positions, sorted_ids.size - 1)
    present = in_bounds & (sorted_ids[safe_positions] == values)
    return present, safe_positions


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def import_malecns(
    sources: MaleCNSSources,
    output: Path,
    *,
    min_weight: int = 5,
) -> MaleCNSImportMetrics:
    """Stream official MaleCNS weights into an atomic canonical snapshot."""

    if min_weight < 1:
        raise ValueError("min_weight must be at least 1")
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"snapshot output already exists: {output}")

    started = time.perf_counter()
    partial.mkdir(parents=True)
    annotations = feather.read_table(sources.annotations)
    neurotransmitters = feather.read_table(sources.neurotransmitters)
    neurons, selection = select_neurons(
        annotations,
        neurotransmitters,
        dataset_id=sources.dataset_id,
    )
    selected_source_annotations = annotations.filter(_valid_annotation_mask(annotations))
    pq.write_table(neurons, partial / "neurons.parquet", compression="zstd")
    pq.write_table(
        selected_source_annotations,
        partial / "source-annotations.parquet",
        compression="zstd",
    )

    selected_ids = neurons.column("neuron_id").to_numpy().astype(np.uint64, copy=False)
    transmitters = neurons.column("transmitter").to_pylist()
    sign_policy = [transmitter_sign(name) for name in transmitters]
    signs_by_position = np.asarray([item[0] for item in sign_policy], dtype=np.int8)
    provenance_by_position = np.asarray([item[1] for item in sign_policy], dtype=object)

    source_weight_rows = 0
    retained_edges = 0
    retained_synapse_weight = 0
    unresolved_sign_edges = 0
    unresolved_sign_weight = 0
    connected_ids: set[int] = set()

    with pa.memory_map(str(sources.weights), "r") as source:
        reader = ipc.open_file(source)
        writer = pq.ParquetWriter(partial / "edges.parquet", EDGE_SCHEMA, compression="zstd")
        try:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                required = {"body_pre", "body_post", "weight"}
                if not required.issubset(batch.schema.names):
                    raise ValueError("weight table requires body_pre, body_post, and weight")
                pre = batch.column(batch.schema.get_field_index("body_pre")).to_numpy().astype(
                    np.uint64, copy=False
                )
                post = batch.column(batch.schema.get_field_index("body_post")).to_numpy().astype(
                    np.uint64, copy=False
                )
                weight = batch.column(batch.schema.get_field_index("weight")).to_numpy().astype(
                    np.int64, copy=False
                )
                source_weight_rows += batch.num_rows
                if np.any(weight <= 0) or np.any(weight > np.iinfo(np.uint32).max):
                    raise ValueError("weight values must fit positive uint32")

                pre_present, pre_positions = _membership_positions(selected_ids, pre)
                post_present, _ = _membership_positions(selected_ids, post)
                keep = (weight >= min_weight) & pre_present & post_present
                if not np.any(keep):
                    continue

                kept_pre = pre[keep]
                kept_post = post[keep]
                kept_weight = weight[keep].astype(np.uint32, copy=False)
                kept_positions = pre_positions[keep]
                kept_signs = signs_by_position[kept_positions]
                kept_provenance = provenance_by_position[kept_positions].tolist()
                kept_count = int(kept_weight.size)
                kept_weight_sum = int(kept_weight.sum(dtype=np.uint64))
                unresolved = kept_signs == 0

                retained_edges += kept_count
                retained_synapse_weight += kept_weight_sum
                unresolved_sign_edges += int(np.count_nonzero(unresolved))
                unresolved_sign_weight += int(kept_weight[unresolved].sum(dtype=np.uint64))
                connected_ids.update(int(value) for value in kept_pre)
                connected_ids.update(int(value) for value in kept_post)

                edge_table = pa.Table.from_arrays(
                    [
                        pa.array(kept_pre, type=pa.uint64()),
                        pa.array(kept_post, type=pa.uint64()),
                        pa.array(kept_weight, type=pa.uint32()),
                        pa.array(kept_signs, type=pa.int8()),
                        pa.array(kept_provenance, type=pa.string()),
                        pa.array(np.full(kept_count, 0.5, dtype=np.float32)),
                    ],
                    schema=EDGE_SCHEMA,
                )
                writer.write_table(edge_table)
        finally:
            writer.close()

    metrics = MaleCNSImportMetrics(
        dataset_id=sources.dataset_id,
        manifest_sha256=sources.manifest_sha256,
        source_annotation_rows=selection.source_annotation_rows,
        source_weight_rows=source_weight_rows,
        selected_neurons=selection.selected_neurons,
        connected_neurons=len(connected_ids),
        untyped_neurons=selection.untyped_neurons,
        missing_transmitters=selection.missing_transmitters,
        retained_edges=retained_edges,
        retained_synapse_weight=retained_synapse_weight,
        unresolved_sign_edges=unresolved_sign_edges,
        unresolved_sign_weight=unresolved_sign_weight,
        min_weight=min_weight,
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    (partial / "metadata.json").write_text(metrics.model_dump_json(indent=2) + "\n")
    partial.replace(output)
    return metrics


def validate_malecns_snapshot(path: Path) -> MaleCNSImportMetrics:
    """Recompute canonical output invariants and compare them with recorded metrics."""

    metrics = MaleCNSImportMetrics.model_validate_json((path / "metadata.json").read_text())
    neurons = pq.read_table(path / "neurons.parquet")
    if neurons.schema != NEURON_SCHEMA:
        raise ValueError("invalid canonical neuron schema")
    neuron_ids = neurons.column("neuron_id").to_numpy().astype(np.uint64, copy=False)
    if neuron_ids.size != np.unique(neuron_ids).size:
        raise ValueError("duplicate canonical neuron ID")
    if neuron_ids.size != metrics.selected_neurons:
        raise ValueError("selected neuron count does not match metadata")

    edge_file = pq.ParquetFile(path / "edges.parquet")
    if edge_file.schema_arrow != EDGE_SCHEMA:
        raise ValueError("invalid canonical edge schema")
    edge_count = 0
    total_weight = 0
    unresolved_edges = 0
    unresolved_weight = 0
    connected_ids: set[int] = set()
    for batch in edge_file.iter_batches():
        pre = batch.column(0).to_numpy().astype(np.uint64, copy=False)
        post = batch.column(1).to_numpy().astype(np.uint64, copy=False)
        weight = batch.column(2).to_numpy().astype(np.uint32, copy=False)
        signs = batch.column(3).to_numpy().astype(np.int8, copy=False)
        if not np.all(_membership_positions(neuron_ids, pre)[0]):
            raise ValueError("edge contains unknown presynaptic neuron")
        if not np.all(_membership_positions(neuron_ids, post)[0]):
            raise ValueError("edge contains unknown postsynaptic neuron")
        if not np.all(np.isin(signs, np.array([-1, 0, 1], dtype=np.int8))):
            raise ValueError("edge contains invalid sign")
        edge_count += batch.num_rows
        total_weight += int(weight.sum(dtype=np.uint64))
        unresolved = signs == 0
        unresolved_edges += int(np.count_nonzero(unresolved))
        unresolved_weight += int(weight[unresolved].sum(dtype=np.uint64))
        connected_ids.update(int(value) for value in pre)
        connected_ids.update(int(value) for value in post)

    observed = (
        edge_count,
        total_weight,
        unresolved_edges,
        unresolved_weight,
        len(connected_ids),
    )
    recorded = (
        metrics.retained_edges,
        metrics.retained_synapse_weight,
        metrics.unresolved_sign_edges,
        metrics.unresolved_sign_weight,
        metrics.connected_neurons,
    )
    if observed != recorded:
        raise ValueError(f"snapshot metrics mismatch: observed={observed}, recorded={recorded}")
    return metrics
