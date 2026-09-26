import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

SNAPSHOT = Path("artifacts/male-cns-v1.0-w5")
REGISTRY = Path("data/registry/hexapod-motor-registry-v2.json")


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="retained MaleCNS unavailable")
def test_cli_runs_retained_foreleg_subtype_controls(tmp_path: Path) -> None:
    output = tmp_path / "foreleg-subtypes.json"
    args = [
        "experiment",
        "foreleg-subtypes",
        str(SNAPSHOT),
        "--registry",
        str(REGISTRY),
        "--output",
        str(output),
        "--steps",
        "30",
        "--seed",
        "7",
        "--drive-interval-steps",
        "10",
        "--drive-amplitude-mv",
        "30",
    ]

    invocation = CliRunner().invoke(app, args)

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["assay"] == "foreleg-subtypes-v1"
    result = payload["result"]
    assert result["drive_interval_steps"] == 10
    assert result["drive_amplitude_mv"] == 30.0
    assert result["graph_unchanged"]
    assert result["behavioral_claim_allowed"] is False
    assert len(result["conditions"]) == 7
    assert sum(item["source_neuron_count"] for item in result["conditions"]) == 58
    assert all(item["replay_exact"] for item in result["conditions"])
    assert all(item["source_lesion_source_spikes"] == 0 for item in result["conditions"])
    assert len(result["mirror_pairs"]) == 3
    assert result["unpaired_subtypes"] == ["hair plate"]
    assert len(result["leave_one_out_controls"]) == 7
    by_excluded = {
        (item["side"], item["excluded_subtype"]): item for item in result["leave_one_out_controls"]
    }
    assert by_excluded[("left", "chordotonal organ")]["source_neuron_count"] == 21
    assert by_excluded[("right", "chordotonal organ")]["source_neuron_count"] == 7
    assert all(
        item["source_lesion_motor_spikes"] == 0 and item["replay_exact"]
        for item in result["leave_one_out_controls"]
    )

    second = CliRunner().invoke(app, args)
    assert second.exit_code != 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
