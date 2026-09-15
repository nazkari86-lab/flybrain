import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mb_association import run_mb_association
from flybrain.schema import EDGE_SCHEMA, NEURON_SCHEMA
from flybrain.shiu import ShiuParameters
from flybrain.shiu_plastic_experiment import (
    build_cue_schedule,
    run_shiu_plastic_integration,
)


def canonical_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({
            "dataset_id": "shiu-fixture",
            "manifest_sha256": "c" * 64,
            "importer": "fixture-importer-v1",
            "min_weight": 5,
        }),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "neuron_id": [1, 2, 3, 4, 10, 11],
                "source_dataset": ["fixture"] * 6,
                "cell_type": ["kc-a", "kc-a", "kc-b", "kc-b", "mbon", "downstream"],
                "superclass": ["central"] * 5 + ["descending"],
                "side": ["L"] * 6,
                "transmitter": ["acetylcholine"] * 6,
                "transmitter_provenance": ["fixture"] * 6,
                "role": ["sensory", "sensory", "sensory", "sensory", "interneuron", "motor"],
                "annotation_status": ["Traced"] * 6,
                "status_label": ["Reviewed"] * 6,
                "annotation_confidence": [1.0] * 6,
            },
            schema=NEURON_SCHEMA,
        ),
        root / "neurons.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "bodyId": [1, 2, 3, 4, 10, 11],
                "class": ["Kenyon_Cell"] * 4 + ["MBON", "DAN"],
                "type": ["KC1", "KC2", "KC3", "KC4", "MBON1", "DAN1"],
            }
        ),
        root / "source-annotations.parquet",
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "pre_id": [1, 2, 3, 4, 1, 2, 10],
                "post_id": [10, 10, 10, 10, 11, 11, 11],
                "synapse_count": [10, 10, 10, 10, 5, 5, 5],
                "sign": [1] * 7,
                "sign_provenance": ["fixture"] * 7,
                "confidence": [1.0] * 7,
            },
            schema=EDGE_SCHEMA,
        ),
        root / "edges.parquet",
    )
    return root


def fixture_graph() -> EventConnectome:
    return EventConnectome.from_sparse(
        SparseConnectome(
            neuron_ids=np.array([1, 2, 3, 4, 10, 11], dtype=np.uint64),
            cell_types=("kc-a", "kc-a", "kc-b", "kc-b", "mbon", "downstream"),
            roles=("sensory",) * 4 + ("interneuron", "motor"),
            adjacency=csr_array(
                (
                    np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32),
                    (np.array([4, 4, 4, 4, 5, 5, 5]), np.array([0, 1, 2, 3, 0, 1, 4])),
                ),
                shape=(6, 6),
            ),
            transmitters=("acetylcholine",) * 6,
            superclasses=("central",) * 5 + ("descending",),
        )
    )


def test_builds_deterministic_staggered_cue_schedule() -> None:
    schedule = build_cue_schedule(
        fixture_graph(),
        np.array([4, 1, 3, 2, 10], dtype=np.uint64),
        ShiuParameters(),
        batch_size=2,
    )

    assert list(schedule.events) == [0, 23, 46]
    assert schedule.steps == 46 + 18 + 250 + 1
    assert schedule.events[0][0].tolist() == [0, 1]
    assert schedule.events[0][1].tolist() == [68.75, 68.75]
    with pytest.raises(ValueError, match="batch_size"):
        build_cue_schedule(
            fixture_graph(), np.array([1], dtype=np.uint64), ShiuParameters(), batch_size=0
        )


def test_paired_shiu_run_measures_dynamic_cue_input_and_replay(tmp_path: Path) -> None:
    snapshot = canonical_snapshot(tmp_path / "snapshot")
    association_state = tmp_path / "association.npz"
    association = run_mb_association(
        snapshot,
        state_path=association_state,
        seed=7,
        cue_size=1,
        trials=2,
    )
    association_path = tmp_path / "association.json"
    association_path.write_text(association.model_dump_json(), encoding="utf-8")

    result = run_shiu_plastic_integration(snapshot, association_path, association_state)

    assert result.passed is True
    assert result.graph_edges == 7
    assert result.matched_plastic_edges == 4
    assert result.cue_a.relative_cue_input_decrease >= 0.10
    assert result.cue_b.relative_cue_input_drift <= 1e-6
    assert result.deterministic_replay_exact is True
    assert result.baseline_graph_unchanged is True
    assert len(result.cue_a.baseline.target_voltage_mv) == result.protocol.steps
    assert len(result.cue_a.learned.target_conductance_mv) == result.protocol.steps
    assert isinstance(result.cue_a.target_spikes_changed, bool)
    assert isinstance(result.cue_a.downstream_spikes_changed, bool)


def test_rejects_association_state_checksum_before_simulation(tmp_path: Path) -> None:
    snapshot = canonical_snapshot(tmp_path / "snapshot")
    association_state = tmp_path / "association.npz"
    association = run_mb_association(snapshot, state_path=association_state, cue_size=1, trials=2)
    association_path = tmp_path / "association.json"
    data = json.loads(association.model_dump_json())
    data["state_sha256"] = "d" * 64
    association_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="state checksum"):
        run_shiu_plastic_integration(snapshot, association_path, association_state)
