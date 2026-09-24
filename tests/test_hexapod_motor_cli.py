import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from test_hexapod_motor_registry import fixture_registry_and_rows
from typer.testing import CliRunner

from flybrain.cli import app

runner = CliRunner()

DN_ROWS = (
    ("d_na02_left", "steering", "DNa02", "L", 5001),
    ("d_na02_right", "steering", "DNa02", "R", 5002),
    ("d_ng13_left", "steering", "DNg13", "L", 5003),
    ("d_ng13_right", "steering", "DNg13", "R", 5004),
    ("mdn_left", "retreat", "MDN", "L", 5005),
    ("mdn_right", "retreat", "MDN", "R", 5006),
)


def cli_fixture(root: Path) -> tuple[Path, Path]:
    registry, rows = fixture_registry_and_rows()
    payload = registry.model_dump(mode="json")
    for name, role, cell_type, side, neuron_id in DN_ROWS:
        rows.append(
            {
                "bodyId": neuron_id,
                "type": cell_type,
                "superclass": "descending_neuron",
                "class": "descending",
                "somaSide": side,
                "rootSide": side,
                "somaNeuromere": "brain",
                "entryNerve": None,
                "exitNerve": None,
            }
        )
        payload["populations"].append(
            {
                "name": name,
                "role": role,
                "selector": {
                    "equals": {
                        "superclass": "descending_neuron",
                        "type": cell_type,
                        "somaSide": side,
                    },
                    "expected_ids": [neuron_id],
                    "expected_count": 1,
                },
                "evidence_ids": ["fixture-annotations"],
            }
        )
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "hexapod-fixture",
                "manifest_sha256": "a" * 64,
                "importer": "hexapod-cli-fixture-v1",
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(rows), snapshot / "source-annotations.parquet")
    neurons = [
        {
            "neuron_id": row["bodyId"],
            "cell_type": row["type"],
            "role": "motor" if row["superclass"] == "vnc_motor" else "sensory",
            "transmitter": "acetylcholine",
            "superclass": row["superclass"],
        }
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(neurons), snapshot / "neurons.parquet")
    proprio_ids = [row["bodyId"] for row in rows if row["superclass"] == "vnc_sensory"]
    motor_ids = [row["bodyId"] for row in rows if row["superclass"] == "vnc_motor"]
    motor_stride = len(motor_ids) // len(proprio_ids)
    edges = [
        {"pre_id": pre, "post_id": post, "synapse_count": 500, "sign": 1}
        for pre, post in zip(proprio_ids, motor_ids[::motor_stride], strict=True)
    ]
    edges.extend(
        {"pre_id": neuron_id, "post_id": motor_ids[index], "synapse_count": 500, "sign": 1}
        for index, (*_, neuron_id) in enumerate(DN_ROWS)
    )
    pq.write_table(pa.Table.from_pylist(edges), snapshot / "edges.parquet")
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    return snapshot, registry_path


def invoke(snapshot: Path, registry: Path, output: Path):
    return runner.invoke(
        app,
        [
            "experiment",
            "hexapod-motor",
            str(snapshot),
            "--registry",
            str(registry),
            "--steps",
            "6",
            "--seed",
            "9",
            "--output",
            str(output),
        ],
    )


def test_cli_atomically_publishes_all_hexapod_assay_families(tmp_path: Path) -> None:
    snapshot, registry = cli_fixture(tmp_path)
    output = tmp_path / "result.json"

    result = invoke(snapshot, registry, output)

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["assay"] == "hexapod-motor-v1"
    assert payload["registry"]["populations"]
    assert set(payload["families"]) == {
        "direct_motor_and_gait",
        "dn_to_motor",
        "proprio_to_motor",
        "closed_loop",
    }
    assert json.loads(result.stdout) == payload


def test_cli_refuses_overwrite_alias_and_snapshot_descendant(tmp_path: Path) -> None:
    snapshot, registry = cli_fixture(tmp_path)
    occupied = tmp_path / "occupied.json"
    occupied.write_text("preserve", encoding="utf-8")

    assert invoke(snapshot, registry, occupied).exit_code != 0
    assert occupied.read_text(encoding="utf-8") == "preserve"
    assert invoke(snapshot, registry, registry).exit_code != 0
    assert invoke(snapshot, registry, snapshot / "result.json").exit_code != 0


def test_cli_identity_failure_leaves_no_output(tmp_path: Path) -> None:
    snapshot, registry = cli_fixture(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["dataset_id"] = "wrong-dataset"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "result.json"

    result = invoke(snapshot, registry, output)

    assert result.exit_code != 0
    assert not output.exists()
    assert not tuple(tmp_path.glob("*.partial"))
