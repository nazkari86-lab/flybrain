import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.flygym_backend import flygym_availability

runner = CliRunner()


@pytest.mark.skipif(
    not flygym_availability().available,
    reason="FlyGym 2.1 physics extra is not installed",
)
def test_practical_autonomy_cli_writes_physics_result_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "practical-autonomy.json"
    video = tmp_path / "practical-autonomy.mp4"
    arguments = [
        "experiment",
        "practical-autonomy",
        "--output",
        str(output),
        "--video",
        str(video),
        "--training-episodes",
        "4",
        "--evaluation-episodes",
        "2",
        "--max-steps",
        "120",
        "--seed",
        "7",
    ]

    first = runner.invoke(app, arguments)

    assert first.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"] == "practical-autonomy-v1"
    assert payload["engineering_demo_only"] is True
    assert payload["physics"]["controlled_joint_dofs"] == 42
    assert payload["physics"]["adhesion_channels"] == 6
    assert payload["physics"]["horizontal_displacement_mm"] > 0.1
    assert payload["physics"]["stable"] is True
    assert payload["physics"]["video_path"] == str(video.resolve())
    assert payload["physics"]["rendered_frames"] >= 2
    assert video.stat().st_size > 1_000
    assert json.loads(first.stdout)["q_table_sha256"] == payload["q_table_sha256"]

    second = runner.invoke(app, arguments)

    assert second.exit_code != 0
    assert "output already exists" in second.output
