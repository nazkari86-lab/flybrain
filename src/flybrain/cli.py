"""Command-line entry points for the Connectome Core milestone."""

import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Literal

import typer

from flybrain.acquire import acquire_artifact
from flybrain.autonomous_hexapod_assay import (
    DEFAULT_LEARNING_REGISTRY,
    DEFAULT_MOTOR_REGISTRY,
    run_retained_autonomous_behavior_benchmark,
    run_retained_autonomous_hexapod_assay,
)
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.bundle import write_bundle
from flybrain.descending_interface import DescendingMap
from flybrain.embodied_episode import EmbodiedEpisodeConfig, run_embodied_episode
from flybrain.embodied_interfaces import MotorMap, SensoryMap
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.experiment import ExperimentConfig, run_experiment
from flybrain.games.cli import games_app
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_benchmark import (
    HexapodBenchmarkThresholds,
    run_hexapod_benchmark,
)
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.hexapod_neural_protocols import (
    run_closed_loop_hexapod_protocol,
    run_dn_to_motor_protocols,
    run_proprio_to_motor_protocols,
)
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.importers.malecns import MaleCNSSources, import_malecns
from flybrain.manifest import load_manifest
from flybrain.mb_association import run_mb_association
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.provenance import snapshot_content_sha256
from flybrain.schema import SnapshotMetadata
from flybrain.shiu_experiment import run_shiu_smoke
from flybrain.shiu_plastic_experiment import run_shiu_plastic_integration
from flybrain.steering_benchmark import run_causal_steering_benchmark

app = typer.Typer(help="Reproducible sparse connectome experiments.")
manifest_app = typer.Typer(help="Validate immutable source declarations.")
data_app = typer.Typer(help="Acquire verified source artifacts.")
snapshot_app = typer.Typer(help="Create canonical sparse snapshots.")
experiment_app = typer.Typer(help="Run and export declared experiments.")
app.add_typer(manifest_app, name="manifest")
app.add_typer(data_app, name="data")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(experiment_app, name="experiment")
app.add_typer(games_app, name="games")


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


@snapshot_app.command("import-malecns")
def import_malecns_snapshot(
    manifest_path: Path,
    cache_root: Annotated[Path, typer.Option("--cache-root")],
    output: Annotated[Path, typer.Option("--output")],
    min_weight: Annotated[int, typer.Option("--min-weight", min=1)] = 5,
) -> None:
    """Import a verified official MaleCNS flat connectome without dense matrices."""

    sources = MaleCNSSources.from_manifest(manifest_path, cache_root)
    metrics = import_malecns(sources, output, min_weight=min_weight)
    typer.echo(metrics.model_dump_json())


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


@experiment_app.command("shiu-smoke")
def shiu_smoke_command(
    snapshot: Path,
    duration_ms: Annotated[float, typer.Option("--duration-ms", min=0.1)] = 10.0,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Run published Shiu dynamics with Poisson drive on all sensory neurons."""

    metrics = run_shiu_smoke(snapshot, duration_ms=duration_ms, seed=seed)
    serialized = metrics.model_dump_json()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(metrics.model_dump_json(indent=2) + "\n")
    typer.echo(serialized)


@experiment_app.command("mb-association")
def mb_association_command(
    snapshot: Path,
    state_output: Annotated[Path, typer.Option("--state-output")],
    output: Annotated[Path, typer.Option("--output")],
    seed: Annotated[int, typer.Option("--seed")] = 7,
    cue_size: Annotated[int, typer.Option("--cue-size", min=1)] = 64,
    trials: Annotated[int, typer.Option("--trials", min=1)] = 3,
    dopamine: Annotated[float, typer.Option("--dopamine", min=0.000001)] = 1.0,
) -> None:
    """Validate persistent cue-specific memory on measured KC-to-MBON edges."""

    state_final = state_output.resolve()
    metrics_final = output.resolve()
    if state_final == metrics_final:
        raise typer.BadParameter("--state-output and --output must be different paths")
    for path in (state_final, metrics_final):
        if path.exists():
            raise typer.BadParameter(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        state_stage = Path(
            stack.enter_context(tempfile.TemporaryDirectory(dir=state_final.parent))
        ) / state_final.name
        metrics_stage = Path(
            stack.enter_context(tempfile.TemporaryDirectory(dir=metrics_final.parent))
        ) / metrics_final.name
        result = run_mb_association(
            snapshot,
            state_path=state_stage,
            seed=seed,
            cue_size=cue_size,
            trials=trials,
            dopamine=dopamine,
        ).model_copy(update={"state_path": str(state_final)})
        metrics_stage.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.link(state_stage, state_final)
        try:
            os.link(metrics_stage, metrics_final)
        except OSError:
            state_final.unlink()
            raise
    typer.echo(result.model_dump_json())


@experiment_app.command("biological-steering")
def biological_steering_command(
    snapshot: Path,
    registry: Annotated[Path, typer.Option("--registry")],
    output: Annotated[Path, typer.Option("--output")],
    steps: Annotated[int, typer.Option("--steps", min=1)] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    """Publish a provenance-bound causal visual steering assay."""

    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    registry_final = registry.resolve()
    if output_final in {snapshot_final, registry_final} or output_final.is_relative_to(
        snapshot_final
    ):
        raise typer.BadParameter(
            "--output must differ from inputs and be outside snapshot"
        )
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")

    declared = load_biological_registry(registry_final)
    resolved = resolve_biological_registry(declared, snapshot_final)
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot_final))
    resolved.validate_graph(graph)
    result = run_causal_steering_benchmark(
        graph,
        resolved,
        steps=steps,
        seed=seed,
        snapshot=str(snapshot_final),
    )

    output_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(result.model_dump_json())


@experiment_app.command("hexapod-motor")
def hexapod_motor_command(
    snapshot: Path,
    registry: Annotated[Path, typer.Option("--registry")],
    output: Annotated[Path, typer.Option("--output")],
    steps: Annotated[int, typer.Option("--steps", min=1)] = 40,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
) -> None:
    """Publish provenance-bound direct, neural, and closed-loop hexapod assays."""

    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    registry_final = registry.resolve()
    if output_final in {snapshot_final, registry_final} or output_final.is_relative_to(
        snapshot_final
    ):
        raise typer.BadParameter(
            "--output must differ from inputs and be outside snapshot"
        )
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")

    declared = load_biological_registry(registry_final)
    resolved = resolve_biological_registry(declared, snapshot_final)
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot_final))
    resolved.validate_graph(graph)
    motor = HexapodMotorMap.from_registry(resolved)
    proprio = ProprioceptiveMap.from_registry(resolved)
    descending = DescendingMap(
        *(resolved.population(name).neuron_ids for name in (
            "d_na02_left",
            "d_na02_right",
            "d_ng13_left",
            "d_ng13_right",
            "mdn_left",
            "mdn_right",
        ))
    )
    direct = run_hexapod_benchmark(
        graph,
        motor,
        steps=steps,
        thresholds=HexapodBenchmarkThresholds(),
    )
    dn = run_dn_to_motor_protocols(
        graph,
        descending,
        motor,
        steps=steps,
        seed=seed,
    )
    proprio_result = run_proprio_to_motor_protocols(
        graph,
        proprio,
        motor,
        steps=steps,
        seed=seed,
    )
    closed_loop = run_closed_loop_hexapod_protocol(
        graph,
        proprio,
        motor,
        body_steps=steps,
        seed=seed,
    )
    payload = {
        "assay": "hexapod-motor-v1",
        "snapshot": str(snapshot_final),
        "steps": steps,
        "seed": seed,
        "registry": resolved.model_dump(mode="json"),
        "families": {
            "direct_motor_and_gait": direct.model_dump(mode="json"),
            "dn_to_motor": dn.model_dump(mode="json"),
            "proprio_to_motor": proprio_result.model_dump(mode="json"),
            "closed_loop": closed_loop.model_dump(mode="json"),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    output_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(serialized)


@experiment_app.command("autonomous-hexapod")
def autonomous_hexapod_command(
    snapshot: Path,
    output: Annotated[Path, typer.Option("--output")],
    learning_registry: Annotated[
        Path, typer.Option("--learning-registry")
    ] = DEFAULT_LEARNING_REGISTRY,
    motor_registry: Annotated[
        Path, typer.Option("--motor-registry")
    ] = DEFAULT_MOTOR_REGISTRY,
    backend: Annotated[
        Literal["reference", "flygym"], typer.Option("--backend")
    ] = "reference",
    steps: Annotated[int, typer.Option("--steps", min=1)] = 2,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
) -> None:
    """Publish a retained schedule-free contact-learning hexapod episode."""

    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    learning_registry_final = learning_registry.resolve()
    motor_registry_final = motor_registry.resolve()
    inputs = {snapshot_final, learning_registry_final, motor_registry_final}
    if output_final in inputs or output_final.is_relative_to(snapshot_final):
        raise typer.BadParameter(
            "--output must differ from inputs and be outside snapshot"
        )
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")
    result = run_retained_autonomous_hexapod_assay(
        snapshot_final,
        learning_registry_path=learning_registry_final,
        motor_registry_path=motor_registry_final,
        backend=backend,
        body_steps=steps,
        seed=seed,
    )
    serialized = result.model_dump_json()
    output_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(serialized)


@experiment_app.command("autonomous-behavior")
def autonomous_behavior_command(
    snapshot: Path,
    output: Annotated[Path, typer.Option("--output")],
    learning_registry: Annotated[
        Path, typer.Option("--learning-registry")
    ] = DEFAULT_LEARNING_REGISTRY,
    motor_registry: Annotated[
        Path, typer.Option("--motor-registry")
    ] = DEFAULT_MOTOR_REGISTRY,
    backend: Annotated[
        Literal["reference", "flygym"], typer.Option("--backend")
    ] = "reference",
    training_episodes: Annotated[int, typer.Option("--training-episodes", min=1)] = 1,
    holdout_episodes: Annotated[int, typer.Option("--holdout-episodes", min=1)] = 1,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 2,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 7,
    seeds: Annotated[str | None, typer.Option("--seeds")] = None,
) -> None:
    """Publish the retained multi-condition autonomous behavior benchmark."""

    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    learning_registry_final = learning_registry.resolve()
    motor_registry_final = motor_registry.resolve()
    inputs = {snapshot_final, learning_registry_final, motor_registry_final}
    if output_final in inputs or output_final.is_relative_to(snapshot_final):
        raise typer.BadParameter(
            "--output must differ from inputs and be outside snapshot"
        )
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")
    parsed_seeds = (
        tuple(int(item.strip()) for item in seeds.split(",") if item.strip())
        if seeds is not None
        else None
    )
    result = run_retained_autonomous_behavior_benchmark(
        snapshot_final,
        learning_registry_path=learning_registry_final,
        motor_registry_path=motor_registry_final,
        backend=backend,
        training_episodes=training_episodes,
        holdout_episodes=holdout_episodes,
        body_steps=steps,
        seed=seed,
        seeds=parsed_seeds,
    )
    output_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(result.model_dump_json())


@experiment_app.command("shiu-plastic")
def shiu_plastic_command(
    snapshot: Path,
    association: Annotated[Path, typer.Option("--association")],
    state: Annotated[Path, typer.Option("--state")],
    output: Annotated[Path, typer.Option("--output")],
    seed: Annotated[int, typer.Option("--seed")] = 7,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
) -> None:
    """Run paired Shiu dynamics with a persisted mushroom-body overlay."""

    output_final = output.resolve()
    input_paths = {snapshot.resolve(), association.resolve(), state.resolve()}
    if output_final in input_paths:
        raise typer.BadParameter("--output must differ from snapshot, association, and state")
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")
    output_final.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        result = run_shiu_plastic_integration(
            snapshot,
            association,
            state,
            seed=seed,
            batch_size=batch_size,
        )
        payload = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
        with stage.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(result.model_dump_json())


@experiment_app.command("embodied-loop")
def embodied_loop_command(
    snapshot: Path,
    output: Annotated[Path, typer.Option("--output")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    sensory_limit: Annotated[int, typer.Option("--sensory-limit", min=1)] = 32,
    motor_limit: Annotated[int, typer.Option("--motor-limit", min=1)] = 16,
) -> None:
    """Run the deterministic world-to-connectome-to-body feedback loop."""

    output_final = output.resolve()
    snapshot_final = snapshot.resolve()
    if output_final == snapshot_final or output_final.is_relative_to(snapshot_final):
        raise typer.BadParameter("--output must be outside snapshot")
    if output_final.exists():
        raise typer.BadParameter(f"output already exists: {output_final}")
    output_final.parent.mkdir(parents=True, exist_ok=True)
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    sensory_ids = tuple(
        int(neuron_id)
        for neuron_id, role in zip(graph.neuron_ids, graph.roles, strict=True)
        if role == "sensory"
    )[:sensory_limit]
    motor_ids = tuple(
        int(neuron_id)
        for neuron_id, role in zip(graph.neuron_ids, graph.roles, strict=True)
        if role == "motor"
    )[:motor_limit]
    if not sensory_ids or not motor_ids:
        raise typer.BadParameter("snapshot must contain sensory and motor neurons")
    config = EmbodiedEpisodeConfig(
        max_steps=max_steps,
        seed=seed,
        sensory_map=SensoryMap(sensory_ids, (), (), ()),
        motor_map=MotorMap((), (), motor_ids),
    )
    world = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(8.0, 5.0),
        threat=(1.0, 1.0),
    )
    metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    source_manifest = metadata.get(
        "source_manifest_sha256", metadata.get("manifest_sha256")
    )
    if not isinstance(metadata.get("dataset_id"), str) or not isinstance(
        source_manifest, str
    ):
        raise typer.BadParameter("snapshot metadata lacks dataset or manifest identity")
    result = run_embodied_episode(
        graph,
        config,
        world=world,
        snapshot=str(snapshot_final),
        dataset_id=metadata["dataset_id"],
        source_manifest_sha256=source_manifest,
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
    )
    with tempfile.TemporaryDirectory(dir=output_final.parent) as temporary:
        stage = Path(temporary) / output_final.name
        with stage.open("xb") as stream:
            stream.write((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, output_final)
    typer.echo(result.model_dump_json())
