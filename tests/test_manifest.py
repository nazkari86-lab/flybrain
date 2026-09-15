import json
from pathlib import Path

import pytest

from flybrain.manifest import load_manifest


def valid_manifest() -> dict[str, object]:
    return {
        "dataset_id": "tiny-v1",
        "source_publication": "https://example.org/paper",
        "license": "CC-BY-4.0",
        "artifacts": [
            {
                "url": "https://example.org/neurons.csv",
                "bytes": 12,
                "sha256": "a" * 64,
            }
        ],
    }


def test_manifest_rejects_artifact_without_sha256(tmp_path: Path) -> None:
    document = valid_manifest()
    del document["artifacts"][0]["sha256"]  # type: ignore[index]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="sha256"):
        load_manifest(path)


def test_manifest_loads_valid_source(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(json.dumps(valid_manifest()))

    manifest = load_manifest(path)

    assert manifest.dataset_id == "tiny-v1"
    assert manifest.artifacts[0].bytes == 12


def test_manifest_rejects_duplicate_artifact_urls(tmp_path: Path) -> None:
    document = valid_manifest()
    document["artifacts"] = [document["artifacts"][0], document["artifacts"][0]]  # type: ignore[index]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="duplicate artifact URL"):
        load_manifest(path)
