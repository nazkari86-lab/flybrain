import math

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.shiu import (
    ShiuParameters,
    ShiuState,
    poisson_voltage_events,
    simulate_shiu,
)


def event_graph(weights: np.ndarray) -> EventConnectome:
    count = weights.shape[0]
    return EventConnectome.from_sparse(
        SparseConnectome(
            neuron_ids=np.arange(1, count + 1, dtype=np.uint64),
            cell_types=tuple("test" for _ in range(count)),
            roles=tuple("test" for _ in range(count)),
            adjacency=csr_array(weights, dtype=np.float32),
        )
    )


def test_analytic_voltage_and_conductance_decay_match_closed_form() -> None:
    params = ShiuParameters()
    graph = event_graph(np.zeros((1, 1), dtype=np.float32))
    state = ShiuState.initial(1, params=params, seed=1)
    state.voltage_mv[0] = -50.0
    state.conductance_mv[0] = 2.0

    list(simulate_shiu(graph, params, steps=1, state=state))

    membrane_decay = math.exp(-params.dt_ms / params.membrane_tau_ms)
    conductance_decay = math.exp(-params.dt_ms / params.conductance_tau_ms)
    expected_g = 2.0 * conductance_decay
    expected_v = params.rest_mv + 2.0 * membrane_decay + 2.0 * (
        params.conductance_tau_ms
        / (params.conductance_tau_ms - params.membrane_tau_ms)
    ) * (conductance_decay - membrane_decay)
    assert math.isclose(float(state.conductance_mv[0]), expected_g, rel_tol=1e-6)
    assert math.isclose(float(state.voltage_mv[0]), expected_v, rel_tol=1e-6)


def test_synaptic_event_arrives_after_exact_published_delay() -> None:
    params = ShiuParameters()
    graph = event_graph(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    state = ShiuState.initial(2, params=params, seed=1)
    events = {0: (np.array([0]), np.array([8.0], dtype=np.float32))}
    simulation = simulate_shiu(graph, params, steps=19, external_voltage_events=events, state=state)

    first = next(simulation)
    assert first.neuron_ids.tolist() == [1]
    for _ in range(17):
        next(simulation)
    assert state.conductance_mv[1] == 0.0
    next(simulation)
    assert state.conductance_mv[1] > 0.0


def test_refractory_period_blocks_repeated_voltage_drive() -> None:
    params = ShiuParameters()
    graph = event_graph(np.zeros((1, 1), dtype=np.float32))
    events = {
        step: (np.array([0]), np.array([8.0], dtype=np.float32)) for step in range(45)
    }

    batches = list(
        simulate_shiu(graph, params, steps=45, external_voltage_events=events, seed=2)
    )

    assert [batch.step for batch in batches if batch.neuron_ids.size] == [0, 22, 44]


def test_poisson_drive_replays_for_same_seed() -> None:
    params = ShiuParameters()
    indices = np.array([1, 3, 5], dtype=np.int64)

    first = poisson_voltage_events(indices, steps=100, params=params, seed=91)
    second = poisson_voltage_events(indices, steps=100, params=params, seed=91)

    assert first.keys() == second.keys()
    for step in first:
        np.testing.assert_array_equal(first[step][0], second[step][0])
        np.testing.assert_array_equal(first[step][1], second[step][1])


def test_silencing_prevents_spike_emission() -> None:
    params = ShiuParameters()
    graph = event_graph(np.zeros((1, 1), dtype=np.float32))
    events = {0: (np.array([0]), np.array([8.0], dtype=np.float32))}

    batches = list(
        simulate_shiu(
            graph,
            params,
            steps=1,
            external_voltage_events=events,
            silenced=np.array([True]),
            seed=4,
        )
    )

    assert batches[0].neuron_ids.size == 0


def test_presynaptic_transmitter_multiplier_scales_outgoing_conductance() -> None:
    graph = EventConnectome(
        neuron_ids=np.array([1, 2], dtype=np.uint64),
        cell_types=("test", "test"),
        roles=("test", "test"),
        transmitters=("acetylcholine", "gaba"),
        superclasses=("test", "test"),
        outgoing=csr_array(np.array([[0.0, 100.0], [0.0, 0.0]], dtype=np.float32)),
    )
    params = ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=0.5)
    events = {0: (np.array([0]), np.array([8.0], dtype=np.float32))}

    normal = ShiuState.initial(2, params=params, seed=2)
    scaled = ShiuState.initial(2, params=params, seed=2)
    list(simulate_shiu(graph, params, steps=2, external_voltage_events=events, state=normal))
    list(
        simulate_shiu(
            graph,
            params,
            steps=2,
            external_voltage_events=events,
            state=scaled,
            presynaptic_transmitter_multipliers={"acetylcholine": 0.5},
        )
    )

    assert scaled.conductance_mv[1] == normal.conductance_mv[1] * 0.5


def test_sparse_plastic_overlay_matches_effective_graph_propagation() -> None:
    graph = event_graph(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [50.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    params = ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=0.5)
    edge_index = int(graph.outgoing.indptr[0])
    multipliers = np.ones(graph.edge_count, dtype=np.float32)
    multipliers[edge_index] = 0.25
    effective = EventConnectome(
        neuron_ids=graph.neuron_ids,
        cell_types=graph.cell_types,
        roles=graph.roles,
        transmitters=graph.transmitters,
        superclasses=graph.superclasses,
        outgoing=graph.outgoing.copy(),
    )
    effective.outgoing.data[edge_index] *= multipliers[edge_index]
    events = {0: (np.array([0]), np.array([8.0], dtype=np.float32))}
    sparse_state = ShiuState.initial(3, params=params, seed=4)
    effective_state = ShiuState.initial(3, params=params, seed=4)
    sparse_batches = list(
        simulate_shiu(
            graph,
            params,
            steps=8,
            external_voltage_events=events,
            state=sparse_state,
            plastic_edge_indices=np.asarray([edge_index], dtype=np.int64),
            plastic_edge_multipliers=np.asarray([0.25], dtype=np.float32),
        )
    )
    effective_batches = list(
        simulate_shiu(
            effective,
            params,
            steps=8,
            external_voltage_events=events,
            state=effective_state,
        )
    )
    assert [batch.neuron_ids.tolist() for batch in sparse_batches] == [
        batch.neuron_ids.tolist() for batch in effective_batches
    ]
    np.testing.assert_array_equal(sparse_state.voltage_mv, effective_state.voltage_mv)
    np.testing.assert_array_equal(
        sparse_state.conductance_mv, effective_state.conductance_mv
    )
    np.testing.assert_array_equal(
        sparse_state.delayed_conductance_mv, effective_state.delayed_conductance_mv
    )


def test_sparse_overlay_is_checked_even_when_no_neuron_fires() -> None:
    graph = event_graph(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="overlay"):
        list(
            simulate_shiu(
                graph,
                ShiuParameters(),
                steps=1,
                plastic_edge_indices=np.array([graph.edge_count], dtype=np.int64),
                plastic_edge_multipliers=np.array([1.0], dtype=np.float32),
            )
        )


def test_poisson_target_can_be_exempt_from_refractory_like_source_model() -> None:
    params = ShiuParameters()
    graph = event_graph(np.zeros((1, 1), dtype=np.float32))
    events = {
        step: (np.array([0]), np.array([8.0], dtype=np.float32)) for step in range(3)
    }

    batches = list(
        simulate_shiu(
            graph,
            params,
            steps=3,
            external_voltage_events=events,
            refractory_exempt=np.array([True]),
            seed=4,
        )
    )

    assert [batch.step for batch in batches if batch.neuron_ids.size] == [0, 1, 2]
