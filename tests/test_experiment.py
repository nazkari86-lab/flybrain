from pathlib import Path

from flybrain.experiment import ExperimentConfig, Stimulus, run_experiment
from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"


def experiment_config(tmp_path: Path, *, lesion: tuple[str, ...] = ()) -> ExperimentConfig:
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
    return ExperimentConfig(
        dataset_id="tiny-v1",
        snapshot_path=snapshot,
        seed=17,
        steps=3,
        dt_ms=1.0,
        tau_ms=1.0,
        rest_mv=0.0,
        reset_mv=0.0,
        threshold_mv=1.0,
        synaptic_scale=1.0,
        stimuli=(Stimulus(step=0, neuron_id=1, current=2.0),),
        lesion_cell_types=lesion,
    )


def test_paired_lesion_reports_causal_motor_spike_difference(tmp_path: Path) -> None:
    result = run_experiment(experiment_config(tmp_path, lesion=("relay",)))

    assert result.motor_spikes == 1
    assert result.lesioned_motor_spikes == 0
    assert result.lesion_effect == 1.0


def test_repeated_experiment_has_identical_declared_observables(tmp_path: Path) -> None:
    config = experiment_config(tmp_path)

    first = run_experiment(config)
    second = run_experiment(config)

    assert first.config_hash == second.config_hash
    assert first.motor_spikes == second.motor_spikes
    assert first.events == second.events
