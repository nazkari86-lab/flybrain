import json
from pathlib import Path

from flybrain.bundle import write_bundle
from flybrain.experiment import ExperimentConfig, Stimulus, run_experiment
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"


def tiny_result(tmp_path: Path):  # type: ignore[no-untyped-def]
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
        seed=5,
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
    return run_experiment(config)


def test_bundle_contains_required_reproducibility_records(tmp_path: Path) -> None:
    bundle = write_bundle(tiny_result(tmp_path), tmp_path / "run")

    assert {path.name for path in bundle.iterdir()} == {
        "config.json",
        "metrics.json",
        "provenance.json",
        "events.jsonl",
        "environment.json",
    }
    metrics = json.loads((bundle / "metrics.json").read_text())
    assert metrics["motor_spikes"] == 1
    assert metrics["lesion_effect"] == 1.0


def test_bundle_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "human-data.txt").write_text("preserve")

    try:
        write_bundle(tiny_result(tmp_path), output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output was overwritten")

    assert (output / "human-data.txt").read_text() == "preserve"
