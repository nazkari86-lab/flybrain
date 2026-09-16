import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()

POPULATIONS = (
    ("visual_r1_r6_left", "sensory", 1),
    ("visual_r1_r6_right", "sensory", 2),
    ("hs_left", "sensory", 3),
    ("hs_right", "sensory", 4),
    ("lc16_left", "sensory", 5),
    ("lc16_right", "sensory", 6),
    ("d_na02_left", "steering", 10),
    ("d_na02_right", "steering", 11),
    ("d_ng13_left", "steering", 20),
    ("d_ng13_right", "steering", 21),
    ("mdn_left", "retreat", 30),
    ("mdn_right", "retreat", 31),
)


def biological_snapshot(root: Path) -> tuple[Path, Path]:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "biological-fixture-v1",
                "manifest_sha256": "a" * 64,
                "importer": "test-fixture-v1",
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "neuron_id": neuron_id,
                    "cell_type": name,
                    "role": "sensory" if role == "sensory" else "interneuron",
                    "transmitter": "acetylcholine",
                    "superclass": "fixture",
                }
                for name, role, neuron_id in POPULATIONS
            ]
        ),
        snapshot / "neurons.parquet",
    )
    edges = ((3, 10), (4, 11), (5, 30), (6, 31), (1, 10), (2, 11))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "pre_id": pre,
                    "post_id": post,
                    "synapse_count": 40,
                    "sign": 1,
                }
                for pre, post in edges
            ]
        ),
        snapshot / "edges.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"bodyId": neuron_id, "type": f"fixture-{name}"}
                for name, _, neuron_id in POPULATIONS
            ]
        ),
        snapshot / "source-annotations.parquet",
    )
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "registry_version": "fixture-v1",
                "dataset_id": "biological-fixture-v1",
                "populations": [
                    {
                        "name": name,
                        "role": role,
                        "selector": {
                            "equals": {"type": f"fixture-{name}"},
                            "expected_ids": [neuron_id],
                            "expected_count": 1,
                        },
                        "evidence_ids": ["fixture-evidence"],
                    }
                    for name, role, neuron_id in POPULATIONS
                ],
                "evidence": [
                    {
                        "evidence_id": "fixture-evidence",
                        "kind": "dataset_measurement",
                        "source_url": "https://example.org/fixture",
                        "claim": "Fixture annotations declare exact test populations.",
                        "confidence": "measured",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return snapshot, registry


def invoke(snapshot: Path, registry: Path, output: Path):
    return runner.invoke(
        app,
        [
            "experiment",
            "biological-steering",
            str(snapshot),
            "--registry",
            str(registry),
            "--steps",
            "40",
            "--output",
            str(output),
        ],
    )


def test_cli_publishes_provenance_bound_biological_result(tmp_path: Path) -> None:
    snapshot, registry = biological_snapshot(tmp_path)
    output = tmp_path / "result.json"
    result = invoke(snapshot, registry, output)
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "biological-steering-v1"
    assert payload["registry"]["populations"]
    assert payload["calibration_passed"] is True
    assert payload["software_revision"] != "source"
    assert set(payload["sensory_claims"]) == {
        "photoreceptor_response",
        "hs_optic_flow",
        "lc16_looming",
        "feature_closed_loop",
    }


def test_cli_refuses_occupied_and_input_alias_outputs(tmp_path: Path) -> None:
    snapshot, registry = biological_snapshot(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("preserve", encoding="utf-8")
    assert invoke(snapshot, registry, output).exit_code != 0
    assert output.read_text(encoding="utf-8") == "preserve"
    assert invoke(snapshot, registry, registry).exit_code != 0
    assert invoke(snapshot, registry, snapshot / "result.json").exit_code != 0


def test_cli_registry_failure_publishes_nothing(tmp_path: Path) -> None:
    snapshot, registry = biological_snapshot(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["populations"][0]["selector"]["expected_ids"] = [999]
    registry.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "result.json"
    result = invoke(snapshot, registry, output)
    assert result.exit_code != 0
    assert not output.exists()
    assert not tuple(tmp_path.glob("*.partial"))
