import os
from pathlib import Path

import pytest

from flybrain.autonomous_hexapod_assay import run_retained_autonomous_hexapod_assay
from flybrain.learning_memory import load_learning_memory, save_learning_memory


def test_retained_learning_can_continue_after_durable_restore(tmp_path: Path) -> None:
    location = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if location is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(location)
    first = run_retained_autonomous_hexapod_assay(snapshot, body_steps=20, seed=7)
    checkpoint = tmp_path / "learning.json"
    save_learning_memory(checkpoint, first.episode.learning_memory)
    restored = load_learning_memory(checkpoint)
    second = run_retained_autonomous_hexapod_assay(
        snapshot, body_steps=20, seed=7, learning_memory=restored,
    )
    assert second.episode.replay_exact
    assert second.episode.graph_unchanged
    assert second.episode.learning_memory.context_sha256 == restored.context_sha256
    assert second.episode.learning_memory.digest != restored.digest
    assert first.episode.routed_dan_spike_events > 0
    assert first.episode.slow_memory_dopamine_effect_max > 0.0
    assert (
        second.episode.slow_memory_dopamine_effect_max
        > first.episode.slow_memory_dopamine_effect_max
    )
