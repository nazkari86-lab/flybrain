import numpy as np
from scipy.sparse import csr_array

from flybrain.dynamics import LIFParameters, simulate_lif
from flybrain.graph import SparseConnectome
from flybrain.interventions import NeuronRecord, silence_mask


def pathway_graph() -> SparseConnectome:
    weights = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32
    )
    return SparseConnectome(
        neuron_ids=np.array([1, 2, 3], dtype=np.uint64),
        cell_types=("sensory", "relay", "motor"),
        roles=("sensory", "interneuron", "motor"),
        adjacency=csr_array(weights),
    )


def run_pathway(mask: np.ndarray) -> int:
    graph = pathway_graph()
    params = LIFParameters(1.0, 1.0, 0.0, 0.0, 1.0)
    inputs = [
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ]
    batches = simulate_lif(graph, params, inputs, seed=3, silenced=mask)
    return sum(3 in batch.neuron_ids for batch in batches)


def test_silencing_relay_removes_motor_spikes() -> None:
    graph = pathway_graph()
    normal = run_pathway(np.zeros(3, dtype=np.bool_))
    lesioned = run_pathway(silence_mask(graph, lambda neuron: neuron.cell_type == "relay"))

    assert normal == 1
    assert lesioned == 0


def test_silence_predicate_receives_stable_neuron_metadata() -> None:
    graph = pathway_graph()

    mask = silence_mask(
        graph,
        lambda neuron: neuron == NeuronRecord(2, "relay", "interneuron"),
    )

    np.testing.assert_array_equal(mask, np.array([False, True, False]))
