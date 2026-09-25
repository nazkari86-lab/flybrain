import numpy as np
import pytest
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


def test_event_graph_accepts_a_distinct_scale_per_fired_source() -> None:
    graph = EventConnectome.from_sparse(
        sparse(
            np.array(
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, -2.0, 0.0]],
                dtype=np.float32,
            )
        )
    )

    output = graph.propagate_indices(
        np.array([0, 1], dtype=np.int64),
        scale=np.array([0.5, 2.0], dtype=np.float32),
    )

    np.testing.assert_allclose(output, np.array([0.0, 2.0, -4.0], dtype=np.float32))


def test_sparse_overlay_rejects_sign_reversal() -> None:
    graph = EventConnectome.from_sparse(
        sparse(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    )
    with pytest.raises(ValueError, match="multiplier"):
        graph.propagate_indices(
            np.array([0], dtype=np.int64),
            scale=1.0,
            edge_indices=np.array([0], dtype=np.int64),
            edge_multipliers=np.array([-0.5], dtype=np.float32),
        )


def test_sparse_overlay_matches_csr_copy_for_large_batch_and_weight_updates() -> None:
    weights = np.zeros((80, 80), dtype=np.float32)
    for pre in range(80):
        weights[(pre + 1) % 80, pre] = float(pre % 5 + 1)
        weights[(pre + 7) % 80, pre] = -float(pre % 3 + 1)
    graph = EventConnectome.from_sparse(sparse(weights))
    canonical = graph.outgoing.data.copy()
    fired = np.arange(70, dtype=np.int64)
    scales = np.linspace(0.25, 1.5, fired.size, dtype=np.float32)
    indices = np.arange(0, graph.edge_count, 7, dtype=np.int64)
    multipliers = np.full(indices.size, 0.5, dtype=np.float32)

    for value in (0.5, 0.0, 1.75):
        multipliers[:] = value
        effective = EventConnectome(
            neuron_ids=graph.neuron_ids,
            cell_types=graph.cell_types,
            roles=graph.roles,
            transmitters=graph.transmitters,
            superclasses=graph.superclasses,
            outgoing=graph.outgoing.copy(),
        )
        effective.outgoing.data[indices] *= multipliers
        actual = graph.propagate_indices(
            fired,
            scale=scales,
            edge_indices=indices,
            edge_multipliers=multipliers,
        )
        expected = effective.propagate_indices(fired, scale=scales)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(graph.outgoing.data, canonical)


def test_event_graph_preserves_neuron_metadata_and_linear_storage() -> None:
    source = sparse(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    graph = EventConnectome.from_sparse(source)

    assert graph.neuron_ids.tolist() == [10, 11]
    assert graph.roles == source.roles
    assert graph.transmitters == source.transmitters
    assert graph.storage_items <= source.neuron_count + 3 * source.edge_count + 1
