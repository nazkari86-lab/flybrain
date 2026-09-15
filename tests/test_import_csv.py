from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flybrain.importers.csv_edges import SnapshotIntegrityError, import_csv_snapshot
from flybrain.schema import EDGE_SCHEMA, NEURON_SCHEMA, SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"


def metadata() -> SnapshotMetadata:
    return SnapshotMetadata(
        dataset_id="tiny-v1",
        source_manifest_sha256="a" * 64,
        importer="csv-edges-v1",
    )


def test_import_rejects_edge_to_unknown_neuron(tmp_path: Path) -> None:
    with pytest.raises(SnapshotIntegrityError, match="unknown neuron 99"):
        import_csv_snapshot(
            FIXTURES / "neurons.csv",
            FIXTURES / "bad_edges.csv",
            tmp_path,
            metadata(),
        )


def test_import_writes_exact_canonical_schemas(tmp_path: Path) -> None:
    snapshot = import_csv_snapshot(
        FIXTURES / "neurons.csv", FIXTURES / "edges.csv", tmp_path, metadata()
    )

    assert pq.read_schema(snapshot / "neurons.parquet") == NEURON_SCHEMA
    assert pq.read_schema(snapshot / "edges.parquet") == EDGE_SCHEMA
    assert (snapshot / "metadata.json").is_file()


def test_import_preserves_unresolved_edge_sign(tmp_path: Path) -> None:
    snapshot = import_csv_snapshot(
        FIXTURES / "neurons.csv",
        FIXTURES / "unresolved_edges.csv",
        tmp_path,
        metadata(),
    )

    edges = pq.read_table(snapshot / "edges.parquet")
    assert edges.column("sign").to_pylist() == [0]
    assert edges.column("sign_provenance").to_pylist() == ["receptor-context-required"]
