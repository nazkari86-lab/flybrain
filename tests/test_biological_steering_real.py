import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()


def test_real_malecns_biological_steering_gate(tmp_path: Path) -> None:
    snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "biological-steering.json"
    result = runner.invoke(
        app,
        [
            "experiment",
            "biological-steering",
            snapshot,
            "--registry",
            "data/registry/biological-interface-registry-v1.json",
            "--steps",
            "40",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["registry"]["dataset_id"] == "male-cns-v1.0-essential"
    assert payload["graph_neurons"] == 166_606
    assert payload["graph_edges"] == 6_240_402
    assert payload["calibration_passed"] is True
    assert payload["replay_exact"] is True
    assert payload["graph_unchanged"] is True
    photoreceptor = payload["sensory_claims"]["photoreceptor_response"]
    assert photoreceptor["classification"] == "null"
    photoreceptor_condition = next(
        item for item in payload["conditions"] if item["name"] == "photoreceptor_left"
    )
    assert photoreceptor_condition["injected_neuron_event_count"] > 0
    assert photoreceptor_condition["upstream_visual_processing_bypassed"] is False
