"""Command-line entry points for the Connectome Core milestone."""

import json
from pathlib import Path
from typing import Annotated

import typer

from flybrain.acquire import acquire_artifact
from flybrain.bundle import write_bundle
from flybrain.experiment import ExperimentConfig, run_experiment
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.manifest import load_manifest
from flybrain.schema import SnapshotMetadata

app = typer.Typer(help="Reproducible sparse connectome experiments.")
manifest_app = typer.Typer(help="Validate immutable source declarations.")
data_app = typer.Typer(help="Acquire verified source artifacts.")
snapshot_app = typer.Typer(help="Create canonical sparse snapshots.")
experiment_app = typer.Typer(help="Run and export declared experiments.")
app.add_typer(manifest_app, name="manifest")
app.add_typer(data_app, name="data")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(experiment_app, name="experiment")


@manifest_app.command("validate")
def validate_manifest(path: Path) -> None:
    """Validate and summarize one JSON source manifest."""

    manifest = load_manifest(path)
    typer.echo(
        json.dumps(
            {"dataset_id": manifest.dataset_id, "artifacts": len(manifest.artifacts)},
            sort_keys=True,
        )
    )


@data_app.command("acquire")
def acquire_data(
    manifest_path: Path,
    root: Annotated[Path, typer.Option("--root")],
) -> None:
    """Acquire every artifact after reserve and checksum validation."""

    manifest = load_manifest(manifest_path)
    for artifact in manifest.artifacts:
        typer.echo(str(acquire_artifact(artifact, root)))


@snapshot_app.command("import-csv")
def import_snapshot(
    neurons: Annotated[Path, typer.Option("--neurons")],
    edges: Annotated[Path, typer.Option("--edges")],
    output: Annotated[Path, typer.Option("--output")],
    dataset_id: Annotated[str, typer.Option("--dataset-id")],
    manifest_sha256: Annotated[str, typer.Option("--manifest-sha256")],
) -> None:
    """Convert canonical-column CSV exports to a sparse snapshot."""

    result = import_csv_snapshot(
        neurons,
        edges,
        output,
        SnapshotMetadata(
            dataset_id=dataset_id,
            source_manifest_sha256=manifest_sha256,
            importer="csv-edges-v1",
        ),
    )
    typer.echo(str(result))


def load_experiment(path: Path) -> ExperimentConfig:
    """Load a complete declarative experiment configuration."""

    return ExperimentConfig.model_validate_json(path.read_text())


@experiment_app.command("run")
def run_command(
    config: Path,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run a paired experiment and write its immutable bundle."""

    result = run_experiment(load_experiment(config))
    bundle = write_bundle(result, output)
    typer.echo(
        json.dumps(
            {
                "motor_spikes": result.motor_spikes,
                "lesioned_motor_spikes": result.lesioned_motor_spikes,
                "lesion_effect": result.lesion_effect,
            },
            sort_keys=True,
        )
    )
    typer.echo(f"bundle={bundle}")
