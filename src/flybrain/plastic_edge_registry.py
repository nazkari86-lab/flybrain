"""Resolve immutable sparse learning-edge manifests from canonical MaleCNS edges."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from flybrain.biological_registry import BiologicalInterfaceRegistry, ResolvedRegistry
from flybrain.schema import EDGE_SCHEMA


class ResolvedPlasticEdgeManifest(BaseModel, frozen=True):
    """One fully measured, deterministic edge manifest; DAN sign zero remains modulatory."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sign: int
    edge_count: int = Field(gt=0)
    contact_count: int = Field(gt=0)
    pre_count: int = Field(gt=0)
    post_count: int = Field(gt=0)
    edge_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _edge_digest(rows: list[tuple[int, int, int, int]]) -> str:
    digest = hashlib.sha256()
    for pre_id, post_id, contacts, sign in rows:
        digest.update(f"{pre_id}\t{post_id}\t{contacts}\t{sign}\n".encode("ascii"))
    return digest.hexdigest()


def resolve_plastic_edge_manifests(
    registry: BiologicalInterfaceRegistry,
    populations: ResolvedRegistry,
    snapshot: Path,
) -> tuple[ResolvedPlasticEdgeManifest, ...]:
    """Stream, validate, and bind only registry-declared edges without dense allocation."""

    resolved_ids = {item.name: frozenset(item.neuron_ids) for item in populations.populations}
    edge_file = pq.ParquetFile(snapshot / "edges.parquet")
    if edge_file.schema_arrow != EDGE_SCHEMA:
        raise ValueError("invalid canonical edge schema")
    results = []
    for declaration in registry.plastic_edge_manifests:
        pre_ids = frozenset().union(*(resolved_ids[name] for name in declaration.pre_populations))
        post_ids = frozenset().union(*(resolved_ids[name] for name in declaration.post_populations))
        rows: list[tuple[int, int, int, int]] = []
        for batch in edge_file.iter_batches(
            columns=["pre_id", "post_id", "synapse_count", "sign"]
        ):
            for pre_id, post_id, contacts, sign in zip(
                batch.column("pre_id").to_pylist(),
                batch.column("post_id").to_pylist(),
                batch.column("synapse_count").to_pylist(),
                batch.column("sign").to_pylist(),
                strict=True,
            ):
                if pre_id in pre_ids and post_id in post_ids:
                    if sign != declaration.sign:
                        raise ValueError(
                            f"plastic edge sign drift for {declaration.name}: {sign}"
                        )
                    rows.append((int(pre_id), int(post_id), int(contacts), int(sign)))
        rows.sort()
        if len(rows) != len({(pre_id, post_id) for pre_id, post_id, _, _ in rows}):
            raise ValueError(f"duplicate plastic edge pair for {declaration.name}")
        edge_count = len(rows)
        contact_count = sum(row[2] for row in rows)
        pre_count = len({row[0] for row in rows})
        post_count = len({row[1] for row in rows})
        digest = _edge_digest(rows)
        expected = (
            edge_count == declaration.expected_edge_count
            and contact_count == declaration.expected_contact_count
            and pre_count == declaration.expected_pre_count
            and post_count == declaration.expected_post_count
            and digest == declaration.expected_edge_sha256
        )
        if not expected:
            raise ValueError(f"plastic edge manifest drift for {declaration.name}")
        results.append(
            ResolvedPlasticEdgeManifest(
                name=declaration.name,
                sign=declaration.sign,
                edge_count=edge_count,
                contact_count=contact_count,
                pre_count=pre_count,
                post_count=post_count,
                edge_sha256=digest,
            )
        )
    return tuple(results)
