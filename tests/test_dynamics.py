import numpy as np
from scipy.sparse import csr_array

from flybrain.dynamics import LIFParameters, LIFState, simulate_lif
from flybrain.graph import SparseConnectome


def graph_with_weights(weights: np.ndarray) -> SparseConnectome:
    count = weights.shape[0]
    return SparseConnectome(
        neuron_ids=np.arange(1, count + 1, dtype=np.uint64),
        cell_types=tuple("test" for _ in range(count)),
        roles=tuple("test" for _ in range(count)),
        adjacency=csr_array(weights, dtype=np.float32),
    )


def currents(*rows: list[float]) -> list[np.ndarray]:
    return [np.asarray(row, dtype=np.float32) for row in rows]


def test_single_neuron_spikes_at_analytic_threshold() -> None:
    params = LIFParameters(
        dt_ms=1.0,
        tau_ms=10.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
    )
    graph = graph_with_weights(np.zeros((1, 1), dtype=np.float32))

    batches = list(simulate_lif(graph, params, currents(*([[2.0]] * 8)), seed=7))

    assert [batch.step for batch in batches if batch.neuron_ids.size] == [6]


def test_recurrent_signal_arrives_one_step_after_presynaptic_spike() -> None:
    graph = graph_with_weights(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    params = LIFParameters(
        dt_ms=1.0,
        tau_ms=1.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
    )

    batches = list(simulate_lif(graph, params, currents([2.0, 0.0], [0.0, 0.0]), seed=1))

    np.testing.assert_array_equal(batches[0].neuron_ids, np.array([1], dtype=np.uint64))
    np.testing.assert_array_equal(batches[1].neuron_ids, np.array([2], dtype=np.uint64))


def test_inhibitory_edge_suppresses_postsynaptic_spike() -> None:
    graph = graph_with_weights(np.array([[0.0, 0.0], [-2.0, 0.0]], dtype=np.float32))
    params = LIFParameters(
        dt_ms=1.0,
        tau_ms=1.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
    )

    batches = list(simulate_lif(graph, params, currents([2.0, 0.0], [0.0, 1.5]), seed=1))

    assert batches[1].neuron_ids.size == 0


def test_simulation_replays_identically_with_same_seed_and_state() -> None:
    graph = graph_with_weights(np.zeros((1, 1), dtype=np.float32))
    params = LIFParameters(
        dt_ms=1.0,
        tau_ms=2.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
    )
    input_rows = currents([0.5], [0.5], [0.5], [0.5])

    first_state = LIFState.initial(graph.neuron_count, seed=19)
    second_state = LIFState.initial(graph.neuron_count, seed=19)
    first = list(simulate_lif(graph, params, input_rows, seed=19, state=first_state))
    second = list(simulate_lif(graph, params, input_rows, seed=19, state=second_state))

    assert [(batch.step, batch.neuron_ids.tolist()) for batch in first] == [
        (batch.step, batch.neuron_ids.tolist()) for batch in second
    ]
    np.testing.assert_array_equal(first_state.voltage, second_state.voltage)
