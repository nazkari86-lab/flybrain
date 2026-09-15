import numpy as np
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome, SparseConnectome


def sparse(weights: np.ndarray) -> SparseConnectome:
    count = weights.shape[0]
    return SparseConnectome(
        neuron_ids=np.arange(10, 10 + count, dtype=np.uint64),
        cell_types=tuple(f"type-{index}" for index in range(count)),
        roles=tuple("sensory" if index == 0 else "interneuron" for index in range(count)),
        adjacency=csr_array(weights, dtype=np.float32),
        transmitters=tuple("acetylcholine" for _ in range(count)),
        superclasses=tuple(
            "cb_sensory" if index == 0 else "cb_intrinsic" for index in range(count)
        ),
    )


def test_event_graph_propagates_in_presynaptic_direction() -> None:
    graph = EventConnectome.from_sparse(
        sparse(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    )

    output = graph.propagate_indices(np.array([0], dtype=np.int64), scale=0.5)

    np.testing.assert_allclose(output, np.array([0.0, 1.0], dtype=np.float32))


def test_event_graph_accumulates_shared_postsynaptic_target() -> None:
    graph = EventConnectome.from_sparse(
        sparse(
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0, 0.0]],
                dtype=np.float32,
            )
        )
    )

    output = graph.propagate_indices(np.array([0, 1], dtype=np.int64), scale=1.0)

    np.testing.assert_allclose(output, np.array([0.0, 0.0, 5.0], dtype=np.float32))


def test_event_graph_preserves_neuron_metadata_and_linear_storage() -> None:
    source = sparse(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    graph = EventConnectome.from_sparse(source)

    assert graph.neuron_ids.tolist() == [10, 11]
    assert graph.roles == source.roles
    assert graph.transmitters == source.transmitters
    assert graph.storage_items <= source.neuron_count + 3 * source.edge_count + 1
