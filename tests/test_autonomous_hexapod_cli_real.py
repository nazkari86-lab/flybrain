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
    assert payload["episode"]["slow_memory_enabled"] is True
    assert payload["episode"]["slow_memory_edges"] == 10_952
    assert payload["episode"]["slow_memory_dopamine_effect_max"] > 0.0
    assert payload["episode"]["slow_memory_nitric_oxide_effect_max"] > 0.0
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
