from pathlib import Path

import numpy as np

from flybrain.graph import SparseConnectome
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"


def build_graph(tmp_path: Path) -> SparseConnectome:
    snapshot = import_csv_snapshot(
        FIXTURES / "neurons.csv",
        FIXTURES / "edges.csv",
        tmp_path,
        SnapshotMetadata(
            dataset_id="tiny-v1",
            source_manifest_sha256="a" * 64,
            importer="csv-edges-v1",
        ),
    )
    return SparseConnectome.from_snapshot(snapshot)


def test_sparse_graph_preserves_direction_and_weights(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)

    propagated = graph.propagate(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    np.testing.assert_array_equal(graph.neuron_ids, np.array([1, 2, 3], dtype=np.uint64))
    np.testing.assert_allclose(propagated, np.array([0.0, 2.0, 0.0], dtype=np.float32))


def test_graph_is_invariant_to_edge_file_order(tmp_path: Path) -> None:
    reversed_edges = tmp_path / "reversed.csv"
    reversed_edges.write_text(
        "pre_id,post_id,synapse_count,sign,sign_provenance,confidence\n"
        "2,3,3,1,fixture,1.0\n"
        "1,2,2,1,fixture,1.0\n"
    )
    first = build_graph(tmp_path / "first")
    second_snapshot = import_csv_snapshot(
        FIXTURES / "neurons.csv",
        reversed_edges,
        tmp_path / "second",
        SnapshotMetadata(
            dataset_id="tiny-v1",
            source_manifest_sha256="a" * 64,
            importer="csv-edges-v1",
        ),
    )
    second = SparseConnectome.from_snapshot(second_snapshot)

    stimulus = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(first.propagate(stimulus), second.propagate(stimulus))


def test_sparse_storage_is_linear_in_neurons_and_edges(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)

    assert graph.storage_items <= graph.neuron_count + 3 * graph.edge_count + 1
