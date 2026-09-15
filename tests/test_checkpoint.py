from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.checkpoint import (
    CheckpointError,
    CheckpointMetadata,
    ConfigHashMismatch,
    load_checkpoint,
    save_checkpoint,
)
from flybrain.dynamics import LIFParameters, LIFState, simulate_lif
from flybrain.graph import SparseConnectome


def one_neuron_graph() -> SparseConnectome:
    return SparseConnectome(
        neuron_ids=np.array([1], dtype=np.uint64),
        cell_types=("test",),
        roles=("test",),
        adjacency=csr_array((1, 1), dtype=np.float32),
    )


def metadata() -> CheckpointMetadata:
    return CheckpointMetadata(config_hash="a" * 64, dataset_id="tiny-v1", seed=11)


def test_checkpoint_rejects_different_configuration(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    save_checkpoint(path, LIFState.initial(1, 11), metadata())

    with pytest.raises(ConfigHashMismatch):
        load_checkpoint(path, expected_config_hash="b" * 64)


def test_checkpoint_rejects_corrupt_archive(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    path.write_bytes(b"not an npz archive")

    with pytest.raises(CheckpointError, match="cannot load checkpoint"):
        load_checkpoint(path, expected_config_hash="a" * 64)


def test_checkpoint_resume_matches_uninterrupted_observables(tmp_path: Path) -> None:
    graph = one_neuron_graph()
    params = LIFParameters(1.0, 2.0, 0.0, 0.0, 1.0)
    rows = [np.array([0.6], dtype=np.float32) for _ in range(4)]

    complete_state = LIFState.initial(1, 11)
    complete = list(simulate_lif(graph, params, rows, seed=11, state=complete_state))

    split_state = LIFState.initial(1, 11)
    before = list(simulate_lif(graph, params, rows[:2], seed=11, state=split_state))
    path = tmp_path / "state.npz"
    save_checkpoint(path, split_state, metadata())
    restored, restored_metadata = load_checkpoint(path, expected_config_hash="a" * 64)
    after = list(simulate_lif(graph, params, rows[2:], seed=11, state=restored))

    assert restored_metadata == metadata()
    assert [(item.step, item.neuron_ids.tolist()) for item in before + after] == [
        (item.step, item.neuron_ids.tolist()) for item in complete
    ]
    np.testing.assert_array_equal(restored.voltage, complete_state.voltage)
    assert not list(tmp_path.glob("*.partial"))
