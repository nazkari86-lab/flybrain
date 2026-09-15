import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.plastic_graph import materialize_plastic_event_graph
from flybrain.plasticity import PlasticEdgeSet


def event_graph(*, zero_first_edge: bool = False) -> EventConnectome:
    first_weight = 0.0 if zero_first_edge else 5.0
    outgoing = csr_array(
        (
            np.array([first_weight, 7.0, -3.0], dtype=np.float32),
            np.array([2, 2, 3], dtype=np.int32),
            np.array([0, 1, 2, 3, 3], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    return EventConnectome(
        neuron_ids=np.array([1, 2, 10, 11], dtype=np.uint64),
        cell_types=("kc-a", "kc-b", "mbon", "downstream"),
        roles=("interneuron", "interneuron", "interneuron", "motor"),
        transmitters=("acetylcholine", "acetylcholine", "gaba", "glutamate"),
        superclasses=("central", "central", "central", "descending"),
        outgoing=outgoing,
    )


def plastic_edges(
    *,
    pre_ids: list[int] | None = None,
    post_ids: list[int] | None = None,
    baseline_weights: list[float] | None = None,
    multipliers: list[float] | None = None,
) -> PlasticEdgeSet:
    edges = PlasticEdgeSet.create(
        pre_ids=np.array(pre_ids if pre_ids is not None else [1, 2], dtype=np.uint64),
        post_ids=np.array(post_ids if post_ids is not None else [10, 10], dtype=np.uint64),
        baseline_weights=np.array(
            baseline_weights if baseline_weights is not None else [5.0, 7.0],
            dtype=np.float32,
        ),
    )
    edges.multipliers[:] = np.array(
        multipliers if multipliers is not None else [0.5, 1.0], dtype=np.float32
    )
    return edges


def csr_arrays(graph: EventConnectome) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        graph.outgoing.data.copy(),
        graph.outgoing.indices.copy(),
        graph.outgoing.indptr.copy(),
    )


def assert_csr_unchanged(
    graph: EventConnectome,
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    np.testing.assert_array_equal(graph.outgoing.data, baseline[0])
    np.testing.assert_array_equal(graph.outgoing.indices, baseline[1])
    np.testing.assert_array_equal(graph.outgoing.indptr, baseline[2])


def test_materializes_only_selected_edges_without_mutating_baseline() -> None:
    graph = event_graph()
    edges = plastic_edges()
    baseline_data = graph.outgoing.data.copy()

    learned, metrics = materialize_plastic_event_graph(graph, edges)

    np.testing.assert_array_equal(graph.outgoing.data, baseline_data)
    assert learned.outgoing[0, 2] == 2.5
    assert learned.outgoing[1, 2] == 7.0
    assert learned.outgoing[2, 3] == -3.0
    assert learned.cell_types is graph.cell_types
    assert learned.roles is graph.roles
    assert learned.transmitters is graph.transmitters
    assert learned.superclasses is graph.superclasses
    assert metrics.matched_edges == 2
    assert metrics.modified_edges == 1
    assert metrics.baseline_absolute_weight == 12.0
    assert metrics.effective_absolute_weight == 9.5
    assert metrics.min_multiplier == 0.5
    assert metrics.max_multiplier == 1.0


def test_rejects_duplicate_plastic_pairs_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = plastic_edges(
        pre_ids=[1, 1],
        post_ids=[10, 10],
        baseline_weights=[5.0, 5.0],
        multipliers=[0.5, 1.0],
    )

    with pytest.raises(ValueError, match="duplicate plastic edge"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_unknown_neuron_ids_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = plastic_edges(pre_ids=[999], post_ids=[10], baseline_weights=[5.0], multipliers=[0.5])

    with pytest.raises(ValueError, match="unknown neuron ID"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_missing_graph_pairs_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = plastic_edges(pre_ids=[1], post_ids=[11], baseline_weights=[5.0], multipliers=[0.5])

    with pytest.raises(ValueError, match="missing graph edge"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_baseline_weight_mismatch_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = plastic_edges(pre_ids=[1], post_ids=[10], baseline_weights=[6.0], multipliers=[0.5])

    with pytest.raises(ValueError, match="baseline weight mismatch"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_zero_sign_graph_edges_without_mutating_baseline() -> None:
    graph = event_graph(zero_first_edge=True)
    baseline = csr_arrays(graph)
    edges = plastic_edges(pre_ids=[1], post_ids=[10], baseline_weights=[5.0], multipliers=[0.5])

    with pytest.raises(ValueError, match="zero-sign graph edge"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_nonfinite_multipliers_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = plastic_edges(pre_ids=[1], post_ids=[10], baseline_weights=[5.0], multipliers=[np.nan])

    with pytest.raises(ValueError, match="multipliers must be finite"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)


def test_rejects_empty_plastic_edge_set_without_mutating_baseline() -> None:
    graph = event_graph()
    baseline = csr_arrays(graph)
    edges = PlasticEdgeSet.create(
        pre_ids=np.array([], dtype=np.uint64),
        post_ids=np.array([], dtype=np.uint64),
        baseline_weights=np.array([], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="at least one plastic edge"):
        materialize_plastic_event_graph(graph, edges)

    assert_csr_unchanged(graph, baseline)
