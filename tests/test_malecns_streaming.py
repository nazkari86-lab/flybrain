from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from flybrain.importers.malecns import (
    MaleCNSSources,
    import_malecns,
    validate_malecns_snapshot,
)


def write_sources(root: Path, *, reverse_weights: bool = False) -> MaleCNSSources:
    root.mkdir(parents=True)
    annotations = pa.table(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["sweet", "relay", "motor", "excluded"],
            "superclass": ["cb_sensory", "cb_intrinsic", "vnc_motor", "tbc_candidate"],
            "somaSide": ["L", "L", "L", "R"],
            "rootSide": ["L", "L", "L", "R"],
            "status": ["Traced"] * 4,
            "statusLabel": ["Reviewed"] * 4,
        }
    )
    neurotransmitters = pa.table(
        {
            "body": [1, 2, 3],
            "consensus_nt": ["acetylcholine", "dopamine", "gaba"],
        }
    )
    rows = [
        (1, 2, 5),
        (2, 3, 7),
        (3, 1, 4),
        (1, 4, 9),
        (4, 1, 9),
        (1, 3, 6),
    ]
    if reverse_weights:
        rows.reverse()
    weights = pa.table(
        {
            "body_pre": [row[0] for row in rows],
            "body_post": [row[1] for row in rows],
            "weight": [row[2] for row in rows],
        }
    )
    stats = pa.table({"body": [1, 2, 3], "pre": [1, 1, 1], "post": [1, 1, 1]})
    paths = {
        "annotations": root / "annotations.feather",
        "neurotransmitters": root / "neurotransmitters.feather",
        "stats": root / "stats.feather",
        "weights": root / "weights.feather",
    }
    feather.write_feather(annotations, paths["annotations"])
    feather.write_feather(neurotransmitters, paths["neurotransmitters"])
    feather.write_feather(stats, paths["stats"])
    feather.write_feather(weights, paths["weights"], chunksize=2)
    return MaleCNSSources(
        annotations=paths["annotations"],
        neurotransmitters=paths["neurotransmitters"],
        stats=paths["stats"],
        weights=paths["weights"],
        manifest_sha256="a" * 64,
        dataset_id="male-cns-fixture",
    )


def normalized_edges(snapshot: Path) -> list[tuple[int, int, int, int]]:
    table = pq.read_table(snapshot / "edges.parquet").sort_by(
        [("pre_id", "ascending"), ("post_id", "ascending")]
    )
    return list(
        zip(
            table.column("pre_id").to_pylist(),
            table.column("post_id").to_pylist(),
            table.column("synapse_count").to_pylist(),
            table.column("sign").to_pylist(),
            strict=True,
        )
    )


def test_streaming_import_accounts_for_threshold_endpoints_and_signs(tmp_path: Path) -> None:
    metrics = import_malecns(write_sources(tmp_path / "source"), tmp_path / "snapshot")

    assert metrics.source_weight_rows == 6
    assert metrics.selected_neurons == 3
    assert metrics.connected_neurons == 3
    assert metrics.retained_edges == 3
    assert metrics.retained_synapse_weight == 18
    assert metrics.unresolved_sign_edges == 1
    assert metrics.unresolved_sign_weight == 7
    assert normalized_edges(tmp_path / "snapshot") == [
        (1, 2, 5, 1),
        (1, 3, 6, 1),
        (2, 3, 7, 0),
    ]
    assert validate_malecns_snapshot(tmp_path / "snapshot") == metrics


def test_streaming_import_is_invariant_to_source_batch_order(tmp_path: Path) -> None:
    first = import_malecns(write_sources(tmp_path / "first"), tmp_path / "first-out")
    second = import_malecns(
        write_sources(tmp_path / "second", reverse_weights=True), tmp_path / "second-out"
    )

    assert first.model_dump(exclude={"runtime_seconds", "peak_rss_bytes"}) == second.model_dump(
        exclude={"runtime_seconds", "peak_rss_bytes"}
    )
    assert normalized_edges(tmp_path / "first-out") == normalized_edges(tmp_path / "second-out")


def test_streaming_import_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "snapshot"
    output.mkdir()
    (output / "human-data.txt").write_text("preserve")

    with pytest.raises(FileExistsError):
        import_malecns(write_sources(tmp_path / "source"), output)

    assert (output / "human-data.txt").read_text() == "preserve"
