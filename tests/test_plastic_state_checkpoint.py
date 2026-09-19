import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.plastic_state_checkpoint import load_plastic_state, save_plastic_state


def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot"
    path.mkdir()
    for name, data in (
        ("metadata.json", b'{}'),
        ("neurons.parquet", b"neurons"),
        ("edges.parquet", b"edges"),
    ):
        (path / name).write_bytes(data)
    return path


def binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1, 3], dtype=np.int64), canonical_edge_count=5
        ),
        pre_ids=np.array([10, 11], dtype=np.uint64),
        post_ids=np.array([20, 21], dtype=np.uint64),
    )


def test_checkpoint_round_trip_is_independent(tmp_path: Path) -> None:
    source = snapshot(tmp_path)
    original = binding()
    original.overlay.multipliers[:] = (0.5, 0.75)
    path = tmp_path / "state.json"

    saved = save_plastic_state(path, original.overlay, source, "a" * 64)
    restored = load_plastic_state(path, original, source, "a" * 64)
    restored.multipliers[0] = 0.2

    assert saved.protocol == "plastic-state-checkpoint-v1"
    assert original.overlay.multipliers.tolist() == [0.5, 0.75]
    assert restored.multipliers.tolist() == pytest.approx([0.2, 0.75])


@pytest.mark.parametrize("registry", ["b" * 64, "not-a-hash"])
def test_checkpoint_rejects_wrong_or_invalid_registry(tmp_path: Path, registry: str) -> None:
    source = snapshot(tmp_path)
    path = tmp_path / "state.json"
    save_plastic_state(path, binding().overlay, source, "a" * 64)
    with pytest.raises(ValueError):
        load_plastic_state(path, binding(), source, registry)


def test_checkpoint_rejects_tampering_and_overwrite(tmp_path: Path) -> None:
    source = snapshot(tmp_path)
    path = tmp_path / "state.json"
    save_plastic_state(path, binding().overlay, source, "a" * 64)
    with pytest.raises(FileExistsError):
        save_plastic_state(path, binding().overlay, source, "a" * 64)
    payload = json.loads(path.read_text())
    payload["multipliers"][0] = 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_plastic_state(path, binding(), source, "a" * 64)
