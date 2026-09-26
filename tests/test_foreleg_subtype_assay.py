import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hexapod_neural_protocols import proprio_fixture

from flybrain.foreleg_subtype_assay import (
    load_foreleg_subtype_labels,
    run_foreleg_subtype_assay,
)


def test_subtype_assay_has_source_lesion_replay_and_homologous_pair() -> None:
    graph, proprio, motor = proprio_fixture(connected=True, target_joint="tibia")

    result = run_foreleg_subtype_assay(
        graph,
        proprio,
        motor,
        subtype_by_id={10: "chordotonal organ", 11: "chordotonal organ"},
        steps=90,
        seed=8,
    )

    assert result["protocol"] == "foreleg-proprio-subtype-open-loop-v1"
    assert result["graph_unchanged"]
    assert len(result["conditions"]) == 2
    left, right = result["conditions"]
    assert left["source_ids"] == (10,)
    assert left["source_spikes"] > 0
    assert left["source_lesion_source_spikes"] == 0
    assert left["source_lesion_motor_spikes"] == 0
    assert left["replay_exact"]
    assert left["tibia_spikes"]["left_fore_tibia_flexor"] > 0
    assert left["motor_spikes_by_population"]["left_fore_tibia_flexor"] > 0
    assert left["motor_spikes"] == sum(left["motor_spikes_by_population"].values())
    assert right["source_ids"] == (11,)
    assert right["tibia_spikes"]["right_fore_tibia_flexor"] > 0
    assert result["mirror_pairs"] == (
        {"subtype": "chordotonal organ", "left_count": 1, "right_count": 1},
    )


def test_subtype_assay_reports_absent_mirror_without_inventing_a_control() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)

    result = run_foreleg_subtype_assay(
        graph,
        proprio,
        motor,
        subtype_by_id={10: "hair plate", 11: "leg"},
        steps=90,
        seed=8,
    )

    assert result["mirror_pairs"] == ()
    assert result["unpaired_subtypes"] == ("hair plate", "leg")


def test_subtype_assay_rejects_missing_foreleg_annotation() -> None:
    graph, proprio, motor = proprio_fixture(connected=True)

    with pytest.raises(ValueError, match="missing foreleg subtype labels"):
        run_foreleg_subtype_assay(
            graph, proprio, motor, subtype_by_id={10: "leg"}, steps=90, seed=8
        )


def test_subtype_assay_does_not_invent_motor_recruitment_on_zero_edge_graph() -> None:
    graph, proprio, motor = proprio_fixture(connected=False)

    result = run_foreleg_subtype_assay(
        graph,
        proprio,
        motor,
        subtype_by_id={10: "leg", 11: "leg"},
        steps=90,
        seed=8,
    )

    assert all(sum(item["tibia_spikes"].values()) == 0 for item in result["conditions"])


def test_subtype_labels_require_correct_retained_sensor_annotation(tmp_path) -> None:
    _, proprio, _ = proprio_fixture(connected=True)
    path = tmp_path / "annotations.parquet"
    data = {
        "bodyId": [10, 11],
        "subclass": ["chordotonal organ", "leg"],
        "superclass": ["vnc_sensory", "vnc_sensory"],
        "class": ["mechanosensory_proprioceptive"] * 2,
        "entryNerve": ["ProLN", "ProLN"],
        "rootSide": ["L", "R"],
    }
    pq.write_table(pa.table(data), path)

    assert load_foreleg_subtype_labels(path, proprio) == {
        10: "chordotonal organ",
        11: "leg",
    }
    data["rootSide"][1] = "L"
    pq.write_table(pa.table(data), path)
    with pytest.raises(ValueError, match="annotation mismatch"):
        load_foreleg_subtype_labels(path, proprio)
