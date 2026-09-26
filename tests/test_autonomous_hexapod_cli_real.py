import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_real_cli_atomically_publishes_autonomous_hexapod_artifact(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "autonomous-hexapod.json"

    invocation = runner.invoke(
        app,
        [
            "experiment",
            "autonomous-hexapod",
            snapshot,
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
    assert payload["protocol"] == "retained-autonomous-hexapod-assay-v1"
    assert payload["graph_neurons"] == 166_606
    assert payload["graph_edges"] == 6_240_402
    assert payload["appetitive_dan_routes"] == 1_211
    assert payload["aversive_dan_routes"] == 188
    assert payload["episode"]["backend"]["name"] == "reference_hexapod"
    assert payload["episode"]["replay_exact"] is True
    assert payload["episode"]["graph_unchanged"] is True
    assert payload["episode"]["motor_spikes"] > 0
    assert (
        payload["episode"]["tactile_contact_model"]
        == "uniform_registered_vnc_tactile_assumption"
    )
    assert payload["episode"]["tactile_contact_events"] > 0
    assert payload["episode"]["slow_memory_enabled"] is True
    assert payload["episode"]["slow_memory_edges"] == 10_952
    assert payload["episode"]["reinforcement_source"] == "contact_gated_neural_dan"
    assert payload["episode"]["final_multipliers"]
    assert payload["software_revision"]
    assert payload["runtime_seconds"] > 0
    assert json.loads(invocation.stdout) == payload

    repeated = runner.invoke(
        app,
        [
            "experiment",
            "autonomous-hexapod",
            snapshot,
            "--steps",
            "2",
            "--output",
            str(output),
        ],
    )
    assert repeated.exit_code != 0


def test_real_cli_can_publish_motor_diagnostics(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "motor-trace.json"

    invocation = runner.invoke(
        app,
        [
            "experiment", "autonomous-hexapod", snapshot,
            "--steps", "2", "--seed", "7", "--backend", "reference",
            "--motor-trace", "--output", str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    trace = payload["episode"]["motor_trace"]
    assert payload["episode"]["replay_exact"] is True
    assert len(trace) == 2
    assert len(trace[0]["activations"]) == 36
    assert trace[-1]["thorax_position_m"] == payload["episode"]["final_body"]["thorax_position_m"]


def test_real_cli_can_publish_explicit_motor_lesion(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "motor-lesion.json"

    invocation = runner.invoke(
        app,
        [
            "experiment", "autonomous-hexapod", snapshot,
            "--steps", "2", "--seed", "7", "--backend", "reference",
            "--motor-trace", "--motor-lesion-group", "left_middle_thorax_coxa_anterior",
            "--output", str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    episode = json.loads(output.read_text(encoding="utf-8"))["episode"]
    assert episode["motor_lesion_groups"] == ["left_middle_thorax_coxa_anterior"]
    assert all(step["activations"][12] == 0.0 for step in episode["motor_trace"])
    assert episode["replay_exact"] is True
    assert episode["graph_unchanged"] is True


def test_real_cli_routes_annotated_proprioception_in_closed_loop(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "annotated-proprioception.json"

    invocation = runner.invoke(
        app,
        [
            "experiment", "autonomous-hexapod", snapshot,
            "--steps", "2", "--seed", "7", "--backend", "reference",
            "--proprioceptive-encoding", "subtype_weighted_spikes",
            "--output", str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    episode = json.loads(output.read_text(encoding="utf-8"))["episode"]
    assert episode["proprioceptive_encoding"] == "subtype_weighted_spikes"
    assert episode["proprioceptive_events"] > 0
    assert episode["replay_exact"] is True
    assert episode["autonomous_behavior_claim_allowed"] is False
