import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_array

from flybrain.biological_registry import (
    AnnotationSelector,
    BiologicalInterfaceRegistry,
    EvidenceRecord,
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.graph import EventConnectome

FIXTURE_REGISTRY = Path(__file__).parent / "fixtures" / "biological_registry_fixture.json"


def annotation_snapshot(
    root: Path,
    rows: list[dict[str, object]],
    *,
    dataset_id: str = "fixture-male-cns",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "manifest_sha256": "a" * 64,
                "importer": "fixture-importer-v1",
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(rows), root / "source-annotations.parquet")
    (root / "neurons.parquet").write_bytes(b"fixture canonical neurons")
    (root / "edges.parquet").write_bytes(b"fixture canonical edges")
    return root


def fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_REGISTRY.read_text(encoding="utf-8"))


def left_dna02(body_id: int = 10) -> dict[str, object]:
    return {
        "bodyId": body_id,
        "type": "DNa02",
        "superclass": "descending_neuron",
        "somaSide": "L",
        "rootSide": None,
        "class": None,
    }


def test_registry_resolves_exact_sorted_population(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(
        tmp_path,
        [
            {
                "bodyId": 30,
                "type": "DNa02",
                "superclass": "descending_neuron",
                "somaSide": "R",
                "rootSide": None,
                "class": None,
            },
            left_dna02(),
        ],
    )
    registry = load_biological_registry(FIXTURE_REGISTRY)

    resolved = resolve_biological_registry(registry, snapshot)

    population = resolved.population("d_na02_left")
    assert population.neuron_ids == (10,)
    assert population.selector.equals["somaSide"] == "L"
    assert population.id_sha256 == hashlib.sha256(b"10").hexdigest()
    assert len(resolved.annotation_sha256) == 64
    assert resolved.source_manifest_sha256 == "a" * 64
    assert len(resolved.snapshot_content_sha256) == 64
    assert tuple(item.evidence_id for item in population.evidence) == (
        "fixture-annotations",
        "d-na02-steering",
    )


@pytest.mark.parametrize("column", ["unknown", "bodyId", "assignedOlHex1"])
def test_selector_rejects_non_allowlisted_columns(column: str) -> None:
    with pytest.raises(ValueError, match="selector column"):
        AnnotationSelector(equals={column: "x"})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"equals": {}, "in_values": {}}, "predicate"),
        (
            {"equals": {"type": "DNa02"}, "in_values": {"type": ("DNa01",)}},
            "conflicting",
        ),
        ({"equals": {"type": "DNa02"}, "expected_ids": (10, 10)}, "expected IDs"),
        ({"equals": {"type": "DNa02"}, "expected_ids": (20, 10)}, "expected IDs"),
    ],
)
def test_selector_rejects_ambiguous_or_unstable_declarations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AnnotationSelector.model_validate(overrides)


@pytest.mark.parametrize("column", ["somaSide", "rootSide"])
def test_selector_rejects_unexpected_laterality(column: str) -> None:
    with pytest.raises(ValueError, match="laterality"):
        AnnotationSelector(equals={column: "X"})


@pytest.mark.parametrize(
    "record",
    [
        {
            "evidence_id": "experiment-without-citation",
            "kind": "experimental_result",
            "source_url": "https://example.org/experiment",
            "claim": "A causal result.",
            "confidence": "high",
        },
        {
            "evidence_id": "assumption-mislabeled",
            "kind": "model_assumption",
            "source_url": "https://example.org/model",
            "claim": "A simulator choice.",
            "confidence": "moderate",
        },
        {
            "evidence_id": "measurement-mislabeled",
            "kind": "dataset_measurement",
            "source_url": "https://example.org/data",
            "claim": "A measured table value.",
            "confidence": "assumption",
        },
    ],
)
def test_evidence_rejects_kind_confidence_contradictions(record: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"evidence|confidence|PMID|DOI"):
        EvidenceRecord.model_validate(record)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate population", "duplicate population"),
        ("duplicate evidence", "duplicate evidence"),
        ("missing evidence", "evidence reference"),
    ],
)
def test_registry_rejects_duplicate_or_unbound_records(mutation: str, message: str) -> None:
    payload = fixture_payload()
    populations = payload["populations"]
    evidence = payload["evidence"]
    assert isinstance(populations, list)
    assert isinstance(evidence, list)
    if mutation == "duplicate population":
        populations.append(deepcopy(populations[0]))
    elif mutation == "duplicate evidence":
        evidence.append(deepcopy(evidence[0]))
    else:
        population = populations[0]
        assert isinstance(population, dict)
        population["evidence_ids"] = ["absent"]

    with pytest.raises(ValueError, match=message):
        BiologicalInterfaceRegistry.model_validate(payload)


def test_registry_rejects_expected_id_drift(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02(11)])

    with pytest.raises(ValueError, match="expected IDs"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_rejects_duplicate_source_body_id(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02(), left_dna02()])

    with pytest.raises(ValueError, match="duplicate source bodyId"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_rejects_missing_annotation_columns(tmp_path: Path) -> None:
    row = left_dna02()
    del row["somaSide"]
    snapshot = annotation_snapshot(tmp_path, [row])

    with pytest.raises(ValueError, match=r"missing annotation columns.*somaSide"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_rejects_empty_population_match(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(
        tmp_path,
        [
            {
                **left_dna02(),
                "bodyId": 30,
                "type": "DNa01",
            }
        ],
    )

    with pytest.raises(ValueError, match=r"population.*empty"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_rejects_nonpositive_source_body_id(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02(0)])

    with pytest.raises(ValueError, match="positive integers"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_rejects_snapshot_dataset_mismatch(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02()], dataset_id="wrong-dataset")

    with pytest.raises(ValueError, match="dataset_id"):
        resolve_biological_registry(load_biological_registry(FIXTURE_REGISTRY), snapshot)


def test_registry_hash_is_stable_across_json_formatting(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path / "snapshot", [left_dna02()])
    reformatted = tmp_path / "registry.json"
    reformatted.write_text(
        json.dumps(fixture_payload(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    original = resolve_biological_registry(
        load_biological_registry(FIXTURE_REGISTRY), snapshot
    )
    alternate = resolve_biological_registry(load_biological_registry(reformatted), snapshot)

    assert original.registry_sha256 == alternate.registry_sha256


def test_resolved_registry_rejects_graph_missing_population_id(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02()])
    resolved = resolve_biological_registry(
        load_biological_registry(FIXTURE_REGISTRY), snapshot
    )
    graph_without_id_10 = EventConnectome(
        neuron_ids=np.array([99], dtype=np.uint64),
        cell_types=("fixture",),
        roles=("interneuron",),
        transmitters=("acetylcholine",),
        superclasses=("cb_intrinsic",),
        outgoing=csr_array((1, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="absent from graph"):
        resolved.validate_graph(graph_without_id_10)


def test_resolved_registry_accepts_graph_with_every_population_id(tmp_path: Path) -> None:
    snapshot = annotation_snapshot(tmp_path, [left_dna02()])
    resolved = resolve_biological_registry(
        load_biological_registry(FIXTURE_REGISTRY), snapshot
    )
    graph = EventConnectome(
        neuron_ids=np.array([10, 99], dtype=np.uint64),
        cell_types=("DNa02", "fixture"),
        roles=("descending", "interneuron"),
        transmitters=("acetylcholine", "acetylcholine"),
        superclasses=("descending_neuron", "cb_intrinsic"),
        outgoing=csr_array((2, 2), dtype=np.float32),
    )

    resolved.validate_graph(graph)
