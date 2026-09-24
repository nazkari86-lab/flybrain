import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app


def test_mbon_descending_cli_publishes_fail_closed_causal_artifact(
    tmp_path: Path,
) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "mbon-descending.json"

    invocation = CliRunner().invoke(
        app,
        [
            "experiment",
            "mbon-descending",
            snapshot,
            "--steps",
            "500",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"] == "retained-mbon-descending-assay-v1"
    assert payload["causal"]["causal_mbon_to_descending_claim_allowed"] is True
    assert payload["causal"]["specific_mbon_dn_motor_claim_allowed"] is False
    assert payload["autonomous_behavior_claim_allowed"] is False
    assert json.loads(invocation.stdout) == payload
