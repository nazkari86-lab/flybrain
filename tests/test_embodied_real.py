import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_real_malecns_embodied_loop_gate(tmp_path: Path) -> None:
    configured = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if configured is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "embodied-malecns.json"

    invocation = runner.invoke(
        app,
        [
            "experiment",
            "embodied-loop",
            configured,
            "--max-steps",
            "100",
            "--output",
            str(output),
        ],
    )

    assert invocation.exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["graph_neurons"] == 166_606
    assert result["graph_edges"] == 6_240_402
    assert result["replay_exact"] is True
    assert result["graph_unchanged"] is True
    assert result["executed_graph_unchanged"] is True
    assert result["passed"] is True
