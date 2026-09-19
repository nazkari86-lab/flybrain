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
    assert payload["protocol"] == "retained-autonomous-behavior-benchmark-v1"
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
    assert benchmark["behavioral_claim_allowed"] is False
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
