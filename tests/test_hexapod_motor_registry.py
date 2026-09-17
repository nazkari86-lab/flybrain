import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_array

from flybrain.biological_registry import (
    AnnotationSelector,
    BiologicalInterfaceRegistry,
    PopulationDeclaration,
    resolve_biological_registry,
)
from flybrain.graph import EventConnectome

REAL_REGISTRY = Path("data/registry/hexapod-motor-registry-v1.json")
RETAINED_MALE_CNS = Path(
    "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5"
)
LEGS = (
    ("left_fore", "L", "T1", "ProLN"),
    ("right_fore", "R", "T1", "ProLN"),
    ("left_middle", "L", "T2", "MesoLN"),
    ("right_middle", "R", "T2", "MesoLN"),
    ("left_hind", "L", "T3", "MetaLN"),
    ("right_hind", "R", "T3", "MetaLN"),
)
JOINT_GROUPS = (
    ("trochanter_flexor", ("Tr flexor MN", "Acc. tr flexor MN")),
    ("trochanter_extensor", ("Tr extensor MN",)),
    ("tibia_flexor", ("Ti flexor MN", "Acc. ti flexor MN")),
    ("tibia_extensor", ("Ti extensor MN",)),
)
EXPECTED_REAL_NAMES = {
    *(f"{leg}_{group}" for leg, *_ in LEGS for group, _ in JOINT_GROUPS),
    *(f"{leg}_proprioception" for leg, *_ in LEGS),
    "d_na02_left",
    "d_na02_right",
    "d_ng13_left",
    "d_ng13_right",
    "mdn_left",
    "mdn_right",
}
EXPECTED_DESCENDING = {
    "d_na02_left": (523769,),
    "d_na02_right": (10360,),
    "d_ng13_left": (11074,),
    "d_ng13_right": (512006,),
    "mdn_left": (11288, 12348),
    "mdn_right": (10763, 11332),
}
EXPECTED_PROPRIOCEPTION = {
    "left_fore_proprioception": (
        41,
        "3a4669ac64ec3481e9dc21862e4547504f8bc8c01d948d7b1154661676974384",
    ),
    "right_fore_proprioception": (
        17,
        "9d4fbd78f11402d05a84dd51786500952ff270f7e63bf84e0f71a297eeed2cdf",
    ),
    "left_middle_proprioception": (
        135,
        "dad088a688069d5dafdf58695e26942588ce9f9dd1ba59505d80d34ab93daf26",
    ),
    "right_middle_proprioception": (
        140,
        "553058e46e38d5c165f6298cb2cf4336ddb8bdf3eff7cfcbc53dc6caa6bb0098",
    ),
    "left_hind_proprioception": (
        138,
        "b104113b764da36fc9c8b26c0818c0acb138b608c0e5588781b6b0367e86dc8a",
    ),
    "right_hind_proprioception": (
        151,
        "9b18767f018616620618bf06a2d761ae85a58a54393e5a06f83d8f89aa211a3e",
    ),
}


def fixture_registry_and_rows() -> tuple[BiologicalInterfaceRegistry, list[dict[str, object]]]:
    populations: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    next_id = 1
    for leg, side, neuromere, nerve in LEGS:
        for group, cell_types in JOINT_GROUPS:
            ids = tuple(range(next_id, next_id + len(cell_types)))
            next_id += len(cell_types)
            for body_id, cell_type in zip(ids, cell_types, strict=True):
                rows.append(
                    {
                        "bodyId": body_id,
                        "type": cell_type,
                        "superclass": "vnc_motor",
                        "class": "motor",
                        "somaSide": side,
                        "rootSide": side,
                        "somaNeuromere": neuromere,
                        "entryNerve": None,
                        "exitNerve": nerve,
                    }
                )
            selector: dict[str, object] = {
                "equals": {
                    "superclass": "vnc_motor",
                    "exitNerve": nerve,
                    "somaSide": side,
                    "somaNeuromere": neuromere,
                },
                "expected_ids": ids,
                "expected_count": len(ids),
            }
            if len(cell_types) == 1:
                selector["equals"]["type"] = cell_types[0]  # type: ignore[index]
            else:
                selector["in_values"] = {"type": cell_types}
            populations.append(
                {
                    "name": f"{leg}_{group}",
                    "role": "motor",
                    "selector": selector,
                    "evidence_ids": ["fixture-annotations"],
                }
            )

        sensory_id = next_id
        next_id += 1
        rows.append(
            {
                "bodyId": sensory_id,
                "type": "leg proprioceptor",
                "superclass": "vnc_sensory",
                "class": "mechanosensory_proprioceptive",
                "somaSide": "R" if side == "L" else "L",
                "rootSide": side,
                "somaNeuromere": neuromere,
                "entryNerve": nerve,
                "exitNerve": None,
            }
        )
        populations.append(
            {
                "name": f"{leg}_proprioception",
                "role": "sensory",
                "selector": {
                    "equals": {
                        "superclass": "vnc_sensory",
                        "class": "mechanosensory_proprioceptive",
                        "entryNerve": nerve,
                        "rootSide": side,
                    },
                    "expected_ids": [sensory_id],
                    "expected_count": 1,
                },
                "evidence_ids": ["fixture-annotations"],
            }
        )

    registry = BiologicalInterfaceRegistry.model_validate(
        {
            "registry_version": "hexapod-fixture-v1",
            "dataset_id": "hexapod-fixture",
            "populations": populations,
            "evidence": [
                {
                    "evidence_id": "fixture-annotations",
                    "kind": "dataset_measurement",
                    "source_url": "https://example.org/hexapod-fixture",
                    "claim": "Fixture annotations declare exact leg populations.",
                    "confidence": "measured",
                }
            ],
        }
    )
    return registry, rows


def write_snapshot(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(
        json.dumps({"dataset_id": "hexapod-fixture", "manifest_sha256": "a" * 64}),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(rows), root / "source-annotations.parquet")
    (root / "neurons.parquet").write_bytes(b"fixture neurons")
    (root / "edges.parquet").write_bytes(b"fixture edges")
    return root


def test_motor_role_and_exact_soma_neuromere_are_strict() -> None:
    declaration = PopulationDeclaration(
        name="left_fore_tibia_flexor",
        role="motor",
        selector=AnnotationSelector(
            equals={"somaNeuromere": "T1", "somaSide": "L"}
        ),
        evidence_ids=("fixture",),
    )

    assert declaration.role == "motor"
    assert declaration.selector.equals["somaNeuromere"] == "T1"
    with pytest.raises(ValueError, match=r"somaNeuromere.*exact"):
        AnnotationSelector(in_values={"somaNeuromere": ("T1", "T2")}, equals={})
    with pytest.raises(ValueError, match=r"somaNeuromere.*string"):
        AnnotationSelector.model_validate({"equals": {"somaNeuromere": 1}})
    with pytest.raises(ValueError, match="role"):
        PopulationDeclaration.model_validate(
            {
                "name": "invalid",
                "role": "locomotion",
                "selector": {"equals": {"type": "Tr flexor MN"}},
                "evidence_ids": ["fixture"],
            }
        )


def test_real_registry_declares_complete_exact_hexapod_interface() -> None:
    registry = BiologicalInterfaceRegistry.model_validate_json(
        REAL_REGISTRY.read_text(encoding="utf-8")
    )
    by_name = {population.name: population for population in registry.populations}

    assert set(by_name) == EXPECTED_REAL_NAMES
    assert sum(
        len(population.selector.expected_ids)
        for population in registry.populations
        if population.role == "motor"
    ) == 148
    for name, population in by_name.items():
        if population.role == "motor":
            assert population.selector.expected_ids
            assert population.selector.equals["superclass"] == "vnc_motor"
            assert set(population.selector.equals) >= {
                "exitNerve",
                "somaSide",
                "somaNeuromere",
                "superclass",
            }
        elif population.role == "sensory":
            count, id_sha256 = EXPECTED_PROPRIOCEPTION[name]
            assert population.selector.expected_count == count
            assert population.selector.expected_id_sha256 == id_sha256
            assert population.selector.equals["rootSide"] in {"L", "R"}
            assert "somaSide" not in population.selector.equals
        else:
            assert population.selector.expected_ids == EXPECTED_DESCENDING[name]
            assert population.selector.equals["superclass"] == "descending_neuron"


def test_real_registry_resolves_retained_malecns_snapshot() -> None:
    if not RETAINED_MALE_CNS.is_dir():
        pytest.skip(f"retained snapshot unavailable: {RETAINED_MALE_CNS}")
    registry = BiologicalInterfaceRegistry.model_validate_json(
        REAL_REGISTRY.read_text(encoding="utf-8")
    )

    resolved = resolve_biological_registry(registry, RETAINED_MALE_CNS)

    assert {population.name for population in resolved.populations} == EXPECTED_REAL_NAMES
    for population in registry.populations:
        actual = resolved.population(population.name)
        if population.role == "motor":
            assert actual.neuron_ids == population.selector.expected_ids
        elif population.role == "sensory":
            expected_count, expected_hash = EXPECTED_PROPRIOCEPTION[population.name]
            assert len(actual.neuron_ids) == expected_count
            assert actual.id_sha256 == expected_hash
        else:
            assert actual.neuron_ids == EXPECTED_DESCENDING[population.name]


def test_fixture_resolves_24_motor_groups_and_six_root_side_banks(tmp_path: Path) -> None:
    registry, rows = fixture_registry_and_rows()
    resolved = resolve_biological_registry(registry, write_snapshot(tmp_path, rows))

    motors = tuple(item for item in resolved.populations if item.role == "motor")
    sensory = tuple(item for item in resolved.populations if item.role == "sensory")
    assert len(motors) == 24
    assert len(sensory) == 6
    assert all("somaNeuromere" in item.selector.equals for item in motors)
    assert all("rootSide" in item.selector.equals for item in sensory)
    assert all("somaSide" not in item.selector.equals for item in sensory)


def test_registry_rejects_overlapping_motor_population(tmp_path: Path) -> None:
    registry, rows = fixture_registry_and_rows()
    duplicate = registry.populations[0].model_copy(update={"name": "duplicate_motor"})
    overlapping = registry.model_copy(update={"populations": (*registry.populations, duplicate)})

    with pytest.raises(ValueError, match="overlapping motor populations"):
        resolve_biological_registry(overlapping, write_snapshot(tmp_path, rows))


@pytest.mark.parametrize("drift", ["side", "count", "ids"])
def test_registry_rejects_motor_snapshot_drift(tmp_path: Path, drift: str) -> None:
    registry, rows = fixture_registry_and_rows()
    if drift == "side":
        rows[0]["somaSide"] = "R"
    elif drift == "count":
        rows.append({**rows[0], "bodyId": 999})
    else:
        rows[0]["bodyId"] = 999

    with pytest.raises(ValueError, match=r"population.*empty|expected IDs|expected count"):
        resolve_biological_registry(registry, write_snapshot(tmp_path, rows))


def test_resolved_hexapod_registry_rejects_missing_graph_motor(tmp_path: Path) -> None:
    registry, rows = fixture_registry_and_rows()
    resolved = resolve_biological_registry(registry, write_snapshot(tmp_path, rows))
    neuron_ids = np.array([row["bodyId"] for row in rows[1:]], dtype=np.uint64)
    size = len(neuron_ids)
    graph = EventConnectome(
        neuron_ids=neuron_ids,
        cell_types=("fixture",) * size,
        roles=("interneuron",) * size,
        transmitters=("acetylcholine",) * size,
        superclasses=("fixture",) * size,
        outgoing=csr_array((size, size), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="absent from graph"):
        resolved.validate_graph(graph)
