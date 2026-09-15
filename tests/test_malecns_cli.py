import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
from typer.testing import CliRunner

from flybrain.acquire import artifact_path
from flybrain.cli import app
from flybrain.manifest import Artifact

runner = CliRunner()


def add_artifact(cache: Path, source: Path, remote_name: str) -> dict[str, object]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = Artifact(
        url=f"https://example.org/{remote_name}",
        bytes=source.stat().st_size,
        sha256=digest,
    )
    target = artifact_path(cache, artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return artifact.model_dump(mode="json")


def fixture_manifest(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    annotations = pa.table(
        {
            "bodyId": [1, 2],
            "type": ["sensory", "motor"],
            "superclass": ["cb_sensory", "vnc_motor"],
            "somaSide": ["L", "L"],
            "rootSide": ["L", "L"],
            "status": ["Traced", "Traced"],
            "statusLabel": ["Reviewed", "Reviewed"],
        }
    )
    tables = {
        "body-annotations-male-cns-v1.0.feather": annotations,
        "body-neurotransmitters-male-cns-v1.0.feather": pa.table(
            {"body": [1, 2], "consensus_nt": ["acetylcholine", "gaba"]}
        ),
        "body-stats-male-cns-v1.0.feather": pa.table({"body": [1, 2]}),
        "connectome-weights-male-cns-v1.0.feather": pa.table(
            {"body_pre": [1], "body_post": [2], "weight": [5]}
        ),
    }
    cache = tmp_path / "cache"
    artifacts = []
    for name, table in tables.items():
        path = raw / name
        feather.write_feather(table, path)
        artifacts.append(add_artifact(cache, path, name))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "male-cns-fixture",
                "source_publication": "https://example.org/paper",
                "license": "CC-BY-4.0",
                "artifacts": artifacts,
            }
        )
    )
    return manifest, cache


def test_cli_imports_verified_malecns_snapshot(tmp_path: Path) -> None:
    manifest, cache = fixture_manifest(tmp_path)
    output = tmp_path / "snapshot"

    result = runner.invoke(
        app,
        [
            "snapshot",
            "import-malecns",
            str(manifest),
            "--cache-root",
            str(cache),
            "--output",
            str(output),
            "--min-weight",
            "5",
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(result.stdout)
    assert metrics["selected_neurons"] == 2
    assert metrics["retained_edges"] == 1
    assert (output / "edges.parquet").is_file()
