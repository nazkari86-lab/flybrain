import numpy as np
from scipy.sparse import csr_array

from flybrain.graph import SparseConnectome


def ring_graph(neuron_count: int, edge_count: int) -> SparseConnectome:
    pre = np.arange(edge_count, dtype=np.int64) % neuron_count
    post = (pre + 1) % neuron_count
    adjacency = csr_array(
        (np.ones(edge_count, dtype=np.float32), (post, pre)),
        shape=(neuron_count, neuron_count),
    )
    adjacency.sum_duplicates()
    return SparseConnectome(
        neuron_ids=np.arange(neuron_count, dtype=np.uint64),
        cell_types=tuple("test" for _ in range(neuron_count)),
        roles=tuple("test" for _ in range(neuron_count)),
        adjacency=adjacency,
    )


def test_sparse_storage_scales_linearly_not_quadratically() -> None:
    small = ring_graph(1_000, 1_000)
    large = ring_graph(10_000, 10_000)

    assert large.storage_bytes < 12 * small.storage_bytes
    assert large.storage_bytes < 1_000_000
