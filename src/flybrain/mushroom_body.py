"""Mushroom-body plastic edge extraction from canonical snapshots."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from flybrain.plasticity import PlasticEdgeSet
from flybrain.provenance import snapshot_sha256
from flybrain.schema import EDGE_SCHEMA


class _SnapshotProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MushroomBodyMetrics(BaseModel, frozen=True):
    """Measured mushroom-body populations and retained plastic anatomy."""

    dataset_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_metadata: dict[str, object]
    kenyon_cells: int
    dopamine_neurons: int
    mbons: int
    plastic_edges: int
    total_synapse_weight: int


def extract_kc_mbon_edges(snapshot: Path) -> tuple[PlasticEdgeSet, MushroomBodyMetrics]:
    """Extract measured KC-to-MBON edges without loading the edge table at once."""

    provenance = _SnapshotProvenance.model_validate_json(
        (snapshot / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = cast(dict[str, object], json.loads(provenance.model_dump_json()))
    annotations = pq.read_table(
        snapshot / "source-annotations.parquet",
        columns=["bodyId", "class", "type"],
    )
    body_ids = [int(value) for value in annotations.column("bodyId").to_pylist()]
    if len(body_ids) != len(set(body_ids)):
        raise ValueError("source annotations contain duplicate bodyId values")

    classes = annotations.column("class").to_pylist()
    kenyon_ids = np.asarray(
        [body_id for body_id, cell_class in zip(body_ids, classes, strict=True)
         if cell_class == "Kenyon_Cell"],
        dtype=np.uint64,
    )
    mbon_ids = np.asarray(
        [body_id for body_id, cell_class in zip(body_ids, classes, strict=True)
         if cell_class == "MBON"],
        dtype=np.uint64,
    )
    dopamine_neurons = sum(cell_class == "DAN" for cell_class in classes)

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    edge_file = pq.ParquetFile(snapshot / "edges.parquet")
    if edge_file.schema_arrow != EDGE_SCHEMA:
        raise ValueError("invalid canonical edge schema")
    for batch in edge_file.iter_batches(columns=["pre_id", "post_id", "synapse_count"]):
        pre_ids = batch.column("pre_id").to_numpy(zero_copy_only=False).astype(
            np.uint64, copy=False
        )
        post_ids = batch.column("post_id").to_numpy(zero_copy_only=False).astype(
            np.uint64, copy=False
        )
        selected = np.isin(pre_ids, kenyon_ids) & np.isin(post_ids, mbon_ids)
        if np.any(selected):
            pre_parts.append(pre_ids[selected])
            post_parts.append(post_ids[selected])
            weight_parts.append(
                batch.column("synapse_count")
                .to_numpy(zero_copy_only=False)
                .astype(np.float32, copy=False)[selected]
            )

    pre_ids = np.concatenate(pre_parts) if pre_parts else np.empty(0, dtype=np.uint64)
    post_ids = np.concatenate(post_parts) if post_parts else np.empty(0, dtype=np.uint64)
    weights = np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=np.float32)
    order = np.lexsort((post_ids, pre_ids))
    edges = PlasticEdgeSet.create(
        pre_ids=pre_ids[order],
        post_ids=post_ids[order],
        baseline_weights=weights[order],
    )
    metrics = MushroomBodyMetrics(
        dataset_id=provenance.dataset_id,
        source_manifest_sha256=provenance.manifest_sha256,
        snapshot_sha256=snapshot_sha256(snapshot),
        snapshot_metadata=metadata,
        kenyon_cells=int(kenyon_ids.size),
        dopamine_neurons=dopamine_neurons,
        mbons=int(mbon_ids.size),
        plastic_edges=int(edges.pre_ids.size),
        total_synapse_weight=int(np.sum(edges.baseline_weights, dtype=np.float64)),
    )
    return edges, metrics
