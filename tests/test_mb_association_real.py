import os
from pathlib import Path

import pytest

from flybrain.mb_association import run_mb_association


def test_real_malecns_snapshot_meets_association_acceptance(tmp_path: Path) -> None:
    configured = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if configured is None:
        pytest.skip("set FLYBRAIN_MALECNS_SNAPSHOT to run the real-data integration test")

    result = run_mb_association(
        Path(configured),
        state_path=tmp_path / "memory.npz",
        seed=7,
        cue_size=64,
        trials=3,
    )

    assert result.passed is True
    assert result.plastic_edges == 33_496
    assert result.target_connected_kcs == 1_502
    assert result.trained_relative_decrease >= result.minimum_trained_relative_decrease
    assert result.untrained_relative_drift <= result.maximum_untrained_relative_drift
