import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flybrain.mushroom_body import extract_kc_mbon_edges
from flybrain.schema import EDGE_SCHEMA


def mushroom_snapshot(root: Path, *, duplicate_body_id: bool = False) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "fixture-male-cns",
                "manifest_sha256": "a" * 64,
                "importer": "fixture-importer-v1",
                "min_weight": 5,
            }
        ),
        encoding="utf-8",
    )
    body_ids = [1, 2, 3, 10, 11, 20]
    if duplicate_body_id:
        body_ids[-1] = 1
    pq.write_table(
        pa.table(
            {
                "bodyId": body_ids,
                "class": ["Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", "MBON", "MBON", "DAN"],
                "type": ["KC1", "KC2", "KC3", "MBON01", "MBON02", "PAM01"],
            }
        ),
        root / "source-annotations.parquet",
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "pre_id": [3, 20, 2, 1, 1],
                "post_id": [11, 10, 10, 2, 10],
                "synapse_count": [9, 11, 7, 13, 5],
                "sign": [1, 1, 1, 0, 1],
                "sign_provenance": ["fixture"] * 5,
                "confidence": [0.5] * 5,
            },
            schema=EDGE_SCHEMA,
        ),
        root / "edges.parquet",
    )
    return root


def test_extracts_only_measured_kc_to_mbon_edges_with_exact_counts(tmp_path: Path) -> None:
    edges, metrics = extract_kc_mbon_edges(mushroom_snapshot(tmp_path / "snapshot"))

    assert metrics.kenyon_cells == 3
    assert metrics.dopamine_neurons == 1
    assert metrics.mbons == 2
    assert metrics.plastic_edges == 3
    assert metrics.total_synapse_weight == 21
    assert metrics.dataset_id == "fixture-male-cns"
    assert metrics.source_manifest_sha256 == "a" * 64
    assert len(metrics.snapshot_sha256) == 64
    assert metrics.snapshot_metadata["importer"] == "fixture-importer-v1"
    assert metrics.snapshot_metadata["min_weight"] == 5
    assert edges.pre_ids.tolist() == [1, 2, 3]
    assert edges.post_ids.tolist() == [10, 10, 11]
    assert edges.baseline_weights.tolist() == [5.0, 7.0, 9.0]


def test_rejects_duplicate_source_body_ids(tmp_path: Path) -> None:
    snapshot = mushroom_snapshot(tmp_path / "snapshot", duplicate_body_id=True)

    with pytest.raises(ValueError, match="duplicate bodyId"):
        extract_kc_mbon_edges(snapshot)


def test_rejects_noncanonical_edge_schema(tmp_path: Path) -> None:
    snapshot = mushroom_snapshot(tmp_path / "snapshot")
    pq.write_table(
        pa.table({"pre_id": [1], "post_id": [10], "synapse_count": [5]}),
        snapshot / "edges.parquet",
    )

    with pytest.raises(ValueError, match="canonical edge schema"):
        extract_kc_mbon_edges(snapshot)
