import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_real_malecns_hexapod_motor_assay(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "hexapod-motor.json"

    invocation = runner.invoke(
        app,
        [
            "experiment",
            "hexapod-motor",
            snapshot,
            "--registry",
            "data/registry/hexapod-motor-registry-v1.json",
            "--steps",
            "90",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    families = payload["families"]
    valid = {"positive", "null", "directionally_wrong", "underpowered"}
    assert payload["registry"]["dataset_id"] == "male-cns-v1.0-essential"
    assert families["direct_motor_and_gait"]["graph_neurons"] == 166_606
    assert families["direct_motor_and_gait"]["graph_edges"] == 6_240_402
    assert families["direct_motor_and_gait"]["calibration_passed"] is True
    assert all(
        claim["classification"] in valid
        for claim in families["dn_to_motor"]["claims"].values()
    )
    assert all(
        claim["classification"] in valid
        for claim in families["proprio_to_motor"]["claims"].values()
    )
    assert families["closed_loop"]["claim"]["classification"] in valid
    assert all(
        
            families[name]["graph_unchanged"] is True
            for name in ("dn_to_motor", "proprio_to_motor", "closed_loop")
        
    )
    assert families["closed_loop"]["sparse_storage_unchanged"] is True
