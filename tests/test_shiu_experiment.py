from pathlib import Path

from flybrain.importers.csv_edges import import_csv_snapshot
from flybrain.schema import SnapshotMetadata
from flybrain.shiu_experiment import run_shiu_smoke

FIXTURES = Path(__file__).parent / "fixtures" / "tiny"


def tiny_snapshot(tmp_path: Path) -> Path:
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


def test_smoke_run_reports_source_parameters_and_deterministic_activity(tmp_path: Path) -> None:
    snapshot = tiny_snapshot(tmp_path)

    first = run_shiu_smoke(snapshot, duration_ms=100.0, seed=7)
    second = run_shiu_smoke(snapshot, duration_ms=100.0, seed=7)

    assert first.stimulated_sensory_neurons == 1
    assert first.external_events > 0
    assert first.parameters["synaptic_delay_ms"] == 1.8
    assert first.parameters["synapse_mv"] == 0.275
    assert first.total_spikes == second.total_spikes
    assert first.reached_neurons == second.reached_neurons
    assert first.spikes_by_role == second.spikes_by_role
