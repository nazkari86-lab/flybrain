import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flybrain.cli import app

SNAPSHOT = Path(
    os.environ.get(
        "FLYBRAIN_MALECNS_SNAPSHOT",
        "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5",
    )
)
REGISTRY = Path("data/registry/biological-interface-registry-v2.json")


def test_visual_looming_cli_publishes_provenance_bound_artifact(tmp_path: Path) -> None:
    if not SNAPSHOT.is_dir():
        pytest.skip(f"retained snapshot unavailable: {SNAPSHOT}")
    output = tmp_path / "looming.json"
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "visual-looming",
            str(SNAPSHOT),
            "--registry",
            str(REGISTRY),
            "--steps",
            "30",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = output.read_text(encoding="utf-8")
    assert '"protocol": "visual-looming-projection-assay-v1"' in payload
    assert '"behavioral_claim_allowed": false' in payload
