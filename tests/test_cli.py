import json
from pathlib import Path

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.experiment import ExperimentConfig, Stimulus
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"
runner = CliRunner()


def test_cli_runs_tiny_lesion_experiment(tmp_path: Path) -> None:
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
    config = ExperimentConfig(
        dataset_id="tiny-v1",
        snapshot_path=snapshot,
        seed=23,
        steps=3,
        dt_ms=1.0,
        tau_ms=1.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
        synaptic_scale=1.0,
        stimuli=(Stimulus(step=0, neuron_id=1, current=2.0),),
        lesion_cell_types=("relay",),
    )
    config_path = tmp_path / "experiment.json"
    config_path.write_text(config.model_dump_json(indent=2))

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path), "--output", str(tmp_path / "bundle")],
    )

    assert result.exit_code == 0
    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["motor_spikes"] == 1
    assert summary["lesion_effect"] == 1.0
    assert (tmp_path / "bundle" / "provenance.json").is_file()


def test_cli_validates_source_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "tiny-v1",
                "source_publication": "https://example.org/paper",
                "license": "CC-BY-4.0",
                "artifacts": [
                    {
                        "url": "https://example.org/a.bin",
                        "bytes": 1,
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )

    result = runner.invoke(app, ["manifest", "validate", str(manifest)])

    assert result.exit_code == 0
    assert "tiny-v1" in result.stdout
