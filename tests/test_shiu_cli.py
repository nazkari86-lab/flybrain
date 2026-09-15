import json
from pathlib import Path

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"
runner = CliRunner()


def test_cli_runs_shiu_smoke_and_writes_metrics(tmp_path: Path) -> None:
    snapshot = import_csv_snapshot(
        FIXTURES / "neurons.csv",
        FIXTURES / "edges.csv",
        tmp_path / "snapshot",
        SnapshotMetadata(
            dataset_id="tiny-v1",
            source_manifest_sha256="a" * 64,
            importer="csv-edges-v1",
        ),
    )
    output = tmp_path / "metrics.json"

    result = runner.invoke(
        app,
        [
            "experiment",
            "shiu-smoke",
            str(snapshot),
            "--duration-ms",
            "100",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(result.stdout)
    assert metrics["stimulated_sensory_neurons"] == 1
    assert metrics["external_events"] > 0
    assert json.loads(output.read_text())["parameters"]["refractory_ms"] == 2.2
