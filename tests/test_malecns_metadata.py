import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from flybrain.acquire import artifact_path
from flybrain.importers.malecns import (
    MaleCNSSources,
    select_neurons,
    transmitter_sign,
)
from flybrain.manifest import Artifact


def cache_artifact(cache_root: Path, name: str, table: pa.Table) -> dict[str, object]:
    temporary = cache_root / name
    temporary.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    artifact = Artifact(
        url=f"https://example.org/{name}",
        bytes=temporary.stat().st_size,
        sha256=digest,
    )
    target = artifact_path(cache_root, artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(target)
    return artifact.model_dump(mode="json")


def test_sources_resolve_verified_content_addressed_artifacts(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    artifacts = [
        cache_artifact(cache_root, "body-annotations-male-cns-v1.0.feather", pa.table({"x": [1]})),
        cache_artifact(
            cache_root, "body-neurotransmitters-male-cns-v1.0.feather", pa.table({"x": [1]})
        ),
        cache_artifact(cache_root, "body-stats-male-cns-v1.0.feather", pa.table({"x": [1]})),
        cache_artifact(
            cache_root, "connectome-weights-male-cns-v1.0.feather", pa.table({"x": [1]})
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "male-cns-v1.0-essential",
                "source_publication": "https://example.org/paper",
                "license": "CC-BY-4.0",
                "artifacts": artifacts,
            }
        )
    )

    sources = MaleCNSSources.from_manifest(manifest_path, cache_root)

    assert sources.annotations.name == artifacts[0]["sha256"]
    assert sources.annotations.is_file()
    assert sources.weights.is_file()


def test_select_neurons_reproduces_valid_superclass_policy_and_fallbacks() -> None:
    annotations = pa.table(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["sweet", None, "excluded", "excluded"],
            "superclass": ["cb_sensory", "vnc_motor", "tbc_candidate", None],
            "somaSide": ["L", None, "R", "L"],
            "rootSide": ["R", "R", "R", "L"],
            "status": ["Traced", "Traced", "Assign", "Orphan"],
            "statusLabel": ["Reviewed", "Roughly traced", "0.5assign", "Orphan"],
        }
    )
    neurotransmitters = pa.table(
        {
            "body": [1, 2, 99],
            "consensus_nt": ["acetylcholine", "dopamine", "gaba"],
        }
    )

    neurons, metrics = select_neurons(annotations, neurotransmitters)

    assert neurons.column("neuron_id").to_pylist() == [1, 2]
    assert neurons.column("cell_type").to_pylist() == ["sweet", "untyped"]
    assert neurons.column("side").to_pylist() == ["L", "R"]
    assert neurons.column("role").to_pylist() == ["sensory", "motor"]
    assert neurons.column("transmitter").to_pylist() == ["acetylcholine", "dopamine"]
    assert metrics.source_annotation_rows == 4
    assert metrics.selected_neurons == 2
    assert metrics.untyped_neurons == 1


def test_transmitter_sign_does_not_invent_modulator_or_glutamate_polarity() -> None:
    assert transmitter_sign("acetylcholine") == (1, "fast-transmitter-assumption")
    assert transmitter_sign("gaba") == (-1, "fast-transmitter-assumption")
    assert transmitter_sign("histamine") == (-1, "fast-transmitter-assumption")
    assert transmitter_sign("glutamate") == (0, "receptor-context-required")
    assert transmitter_sign("dopamine") == (0, "neuromodulator-not-fast-sign")
    assert transmitter_sign("unclear") == (0, "unresolved-transmitter")
