import json
from dataclasses import replace
from pathlib import Path

import pytest

from flybrain.plasticity import PlasticStateIdentity
from flybrain.provenance import (
    snapshot_content_sha256,
    snapshot_sha256,
    validate_state_identity,
)


def minimal_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "fixture",
                "manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (root / "source-annotations.parquet").write_bytes(b"source annotations")
    (root / "edges.parquet").write_bytes(b"edges")
    return root


def test_snapshot_identity_is_deterministic_and_changes_with_content(tmp_path: Path) -> None:
    snapshot = minimal_snapshot(tmp_path / "snapshot")

    first = snapshot_sha256(snapshot)

    assert first == snapshot_sha256(snapshot)
    (snapshot / "metadata.json").write_text('{"changed": true}', encoding="utf-8")
    assert snapshot_sha256(snapshot) != first


def test_validates_plastic_state_against_snapshot_identity(tmp_path: Path) -> None:
    snapshot = minimal_snapshot(tmp_path / "snapshot")
    identity = PlasticStateIdentity(
        dataset_id="fixture",
        source_manifest_sha256="a" * 64,
        snapshot_sha256=snapshot_sha256(snapshot),
    )

    validate_state_identity(snapshot, identity)

    with pytest.raises(ValueError, match="snapshot identity"):
        validate_state_identity(snapshot, replace(identity, snapshot_sha256="b" * 64))


def test_content_digest_requires_each_core_snapshot_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "neurons.parquet").write_bytes(b"neurons")
    (snapshot / "edges.parquet").write_bytes(b"edges")
    (snapshot / "source-annotations.parquet").write_bytes(b"annotations")

    with pytest.raises(ValueError, match="metadata, neurons, and edges"):
        snapshot_content_sha256(snapshot)
