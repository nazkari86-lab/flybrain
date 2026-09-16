import json
from pathlib import Path

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"
runner = CliRunner()


def snapshot(tmp_path: Path) -> Path:
    return import_csv_snapshot(
        FIXTURES / "neurons.csv",
        FIXTURES / "edges.csv",
        tmp_path / "snapshot",
        SnapshotMetadata(
            dataset_id="tiny-v1",
            source_manifest_sha256="a" * 64,
            importer="csv-edges-v1",
        ),
    )


def invoke(snapshot_path: Path, output: Path):
    return runner.invoke(
        app,
        [
            "experiment",
            "embodied-loop",
            str(snapshot_path),
            "--max-steps",
            "8",
            "--output",
            str(output),
        ],
    )


def test_cli_publishes_embodied_loop_result(tmp_path: Path) -> None:
    snapshot_path = snapshot(tmp_path)
    output = tmp_path / "embodied.json"

    result = invoke(snapshot_path, output)

    assert result.exit_code == 0
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["benchmark"] == "embodied-loop-v1"
    assert metrics["replay_exact"] is True
    assert metrics["steps"] == 8


def test_cli_refuses_occupied_output_and_input_alias(tmp_path: Path) -> None:
    snapshot_path = snapshot(tmp_path)
    output = tmp_path / "embodied.json"
    output.write_text("preserve", encoding="utf-8")

    occupied = invoke(snapshot_path, output)
    aliased = invoke(snapshot_path, snapshot_path)

    assert occupied.exit_code != 0
    assert output.read_text(encoding="utf-8") == "preserve"
    assert aliased.exit_code != 0


def test_cli_refuses_output_inside_snapshot(tmp_path: Path) -> None:
    snapshot_path = snapshot(tmp_path)

    result = invoke(snapshot_path, snapshot_path / "result.json")

    assert result.exit_code != 0
    assert not (snapshot_path / "result.json").exists()
