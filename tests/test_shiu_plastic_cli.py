import json
from pathlib import Path

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.mb_association import run_mb_association
from test_shiu_plastic_experiment import canonical_snapshot

runner = CliRunner()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot = canonical_snapshot(tmp_path / "snapshot")
    state = tmp_path / "association.npz"
    association = run_mb_association(snapshot, state_path=state, cue_size=1, trials=2)
    association_path = tmp_path / "association.json"
    association_path.write_text(association.model_dump_json(), encoding="utf-8")
    return snapshot, association_path, state


def test_cli_writes_shiu_plastic_result_atomically(tmp_path: Path) -> None:
    snapshot, association, state = _inputs(tmp_path)
    output = tmp_path / "result.json"

    result = runner.invoke(
        app,
        [
            "experiment",
            "shiu-plastic",
            str(snapshot),
            "--association",
            str(association),
            "--state",
            str(state),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_cli_preserves_occupied_output(tmp_path: Path) -> None:
    snapshot, association, state = _inputs(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("keep", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "experiment",
            "shiu-plastic",
            str(snapshot),
            "--association",
            str(association),
            "--state",
            str(state),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_rejects_output_equal_to_input(tmp_path: Path) -> None:
    snapshot, association, state = _inputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "experiment",
            "shiu-plastic",
            str(snapshot),
            "--association",
            str(association),
            "--state",
            str(state),
            "--output",
            str(state),
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "state").exists()
