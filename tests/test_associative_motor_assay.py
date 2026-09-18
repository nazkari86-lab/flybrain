import os
from pathlib import Path

import pytest

from flybrain.associative_motor_assay import run_retained_associative_motor_assay


def test_retained_assay_binds_anatomy_learning_body_and_controls() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")

    result = run_retained_associative_motor_assay(Path(raw_snapshot), body_steps=1)

    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.graph_neurons == 166_606
    assert result.graph_edges == 6_240_402
    assert result.anatomy.direct_edge_count == 0
    assert result.anatomy.two_hop_edge_count == 31
    assert result.paired_training.replay_exact is True
    assert result.paired_training.graph_unchanged is True
    assert result.motor_lesion.motor_spikes == 0
    assert result.no_contact_preserves_overlay is True
