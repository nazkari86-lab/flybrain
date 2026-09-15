import os
from pathlib import Path

import pytest

from flybrain.shiu_plastic_experiment import run_shiu_plastic_integration


def test_real_malecns_shiu_plastic_gate() -> None:
    names = (
        "FLYBRAIN_MALECNS_SNAPSHOT",
        "FLYBRAIN_MB_ASSOCIATION_JSON",
        "FLYBRAIN_MB_STATE_NPZ",
    )
    values = [os.environ.get(name) for name in names]
    if any(value is None for value in values):
        pytest.skip("real MaleCNS inputs are not configured")

    result = run_shiu_plastic_integration(*(Path(value) for value in values if value is not None))
    assert result.graph_neurons == 166_606
    assert result.graph_edges == 6_240_402
    assert result.matched_plastic_edges == 33_496
    assert result.passed is True
    assert result.deterministic_replay_exact is True
    assert result.baseline_graph_unchanged is True
