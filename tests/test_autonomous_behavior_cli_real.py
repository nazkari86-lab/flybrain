import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_real_behavior_cli_publishes_all_controls_atomically(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "behavior.json"
    invocation = runner.invoke(
        app,
        [
            "experiment",
            "autonomous-behavior",
            snapshot,
            "--training-episodes",
            "1",
            "--holdout-episodes",
            "1",
            "--steps",
            "2",
            "--seed",
            "7",
            "--backend",
            "reference",
            "--output",
            str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    benchmark = payload["benchmark"]
    assert payload["protocol"] == "retained-autonomous-behavior-benchmark-v2"
    assert payload["graph_neurons"] == 166_606
    assert payload["graph_edges"] == 6_240_402
    assert set(benchmark["condition_observations"]) == {
        "normal",
        "no_plasticity",
        "dan_lesion",
        "kc_mbon_lesion",
        "rewired_control",
    }
    assert benchmark["graph_unchanged"] is True
    assert benchmark["evidence_protocol"] == "measured-replay-persistent-memory-v6"
    assert benchmark["generalization_verified"] is True
    assert benchmark["replay_exact"] is True
    assert benchmark["holdout_weights_frozen"] is True
    assert benchmark["holdout_memory_frozen"] is True
    assert benchmark["unassisted_motor_output"] is True
    assert benchmark["rewired_control_effective"] is True
    assert len(benchmark["rewired_control_evidence"]) == 1
    assert benchmark["rewired_control_evidence"][0]["changed_edges"] > 0
    assert len(benchmark["episode_evidence"]) == 30
    assert all(x["replay_performed"] and x["replay_exact"] for x in benchmark["episode_evidence"])
    assert all(
        x["weights_unchanged"] and not x["plasticity_enabled"]
        for x in benchmark["episode_evidence"] if x["holdout"]
    )
    assert all(
        sum(x["dan_spike_counts"].values()) == 0
        for x in benchmark["holdout_neural_activity"]["dan_lesion"]
    )
    assert benchmark["behavioral_claim_allowed"] is False
    assert benchmark["odor_channel_model"] == "measured_bilateral_receptor_channels"
    assert benchmark["odor_assignment"]["food_cell_types"] == ["ORN_DM1", "ORN_VA2"]
    assert benchmark["odor_assignment"]["threat_cell_types"] == ["ORN_DA2"]
    assert benchmark["odor_assignment"]["food_evidence_doi"] == "10.1038/nature07983"
    assert (
        benchmark["odor_assignment"]["threat_evidence_doi"]
        == "10.1016/j.cell.2012.09.046"
    )
    assert len(benchmark["condition_observations"]["normal"]) == 4
    assert benchmark["training_summaries"]["normal"]["appetitive_contacts"] > 0
    assert benchmark["training_summaries"]["normal"]["aversive_contacts"] > 0
    assert benchmark["training_summaries"]["normal"]["dan_events"] > 0
    assert benchmark["training_summaries"]["dan_lesion"]["dan_events"] == 0
    descending = benchmark["holdout_neural_activity"]["normal"][0]["descending_spikes"]
    assert set(descending) == {
        "d_na02_left",
        "d_na02_right",
        "d_ng13_left",
        "d_ng13_right",
        "mdn_left",
        "mdn_right",
    }
    assert all(value >= 0 for value in descending.values())
    assert json.loads(invocation.stdout) == payload

    repeated = runner.invoke(
        app,
        [
            "experiment",
            "autonomous-behavior",
            snapshot,
            "--output",
            str(output),
        ],
    )
    assert repeated.exit_code != 0


def test_real_behavior_cli_keeps_subtype_encoder_in_all_control_episodes(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "subtype-behavior.json"
    invocation = runner.invoke(
        app,
        [
            "experiment", "autonomous-behavior", snapshot,
            "--training-episodes", "1", "--holdout-episodes", "1",
            "--steps", "2", "--seed", "7", "--backend", "reference",
            "--proprioceptive-encoding", "subtype_weighted_spikes",
            "--output", str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    benchmark = json.loads(output.read_text(encoding="utf-8"))["benchmark"]
    assert benchmark["proprioceptive_encoding"] == "subtype_weighted_spikes"
    assert benchmark["replay_exact"] is True
    assert benchmark["holdout_weights_frozen"] is True
    assert len(benchmark["episode_evidence"]) == 30
    assert benchmark["behavioral_claim_allowed"] is False
