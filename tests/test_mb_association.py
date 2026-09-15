import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.mb_association import run_mb_association
from flybrain.schema import EDGE_SCHEMA

runner = CliRunner()


def association_snapshot(root: Path, *, duplicate_bias: bool = False) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"dataset_id": "association-fixture", "manifest_sha256": "b" * 64}),
        encoding="utf-8",
    )
    body_ids = [1, 2, 3, 4, 5, 6, 10, 11, 20]
    pq.write_table(
        pa.table(
            {
                "bodyId": body_ids,
                "class": ["Kenyon_Cell"] * 6 + ["MBON", "MBON", "DAN"],
                "type": ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6", "M1", "M2", "D1"],
            }
        ),
        root / "source-annotations.parquet",
    )
    if duplicate_bias:
        pre_ids = [1, 1, 2, 2, 3, 4, 5]
        post_ids = [10, 10, 10, 10, 11, 11, 11]
        weights = [10, 10, 10, 10, 10, 10, 10]
    else:
        pre_ids = [1, 2, 3, 4, 5, 6, 1, 2, 20]
        post_ids = [10, 10, 10, 10, 10, 10, 11, 11, 10]
        weights = [10, 11, 12, 13, 14, 15, 8, 9, 20]
    pq.write_table(
        pa.Table.from_pydict(
            {
                "pre_id": pre_ids,
                "post_id": post_ids,
                "synapse_count": weights,
                "sign": [1] * len(pre_ids),
                "sign_provenance": ["fixture"] * len(pre_ids),
                "confidence": [1.0] * len(pre_ids),
            },
            schema=EDGE_SCHEMA,
        ),
        root / "edges.parquet",
    )
    return root


def test_real_topology_benchmark_is_cue_specific_controlled_and_persistent(
    tmp_path: Path,
) -> None:
    snapshot = association_snapshot(tmp_path / "snapshot")
    state = tmp_path / "memory.npz"

    result = run_mb_association(
        snapshot,
        state_path=state,
        seed=17,
        cue_size=2,
        trials=3,
    )

    assert result.target_mbon_id == 10
    assert result.target_connected_kcs == 6
    assert len(result.cue_a_ids) == len(result.cue_b_ids) == 2
    assert set(result.cue_a_ids).isdisjoint(result.cue_b_ids)
    assert result.trained_relative_decrease >= 0.10
    assert result.untrained_relative_drift < 1e-6
    assert result.no_dopamine_relative_drift == 0.0
    assert result.cleared_eligibility_relative_drift == 0.0
    assert result.control_trials == result.trials
    assert result.persistence_replay_exact is True
    assert result.state_sha256 == hashlib.sha256(state.read_bytes()).hexdigest()

    replay = run_mb_association(
        snapshot,
        state_path=tmp_path / "memory-replay.npz",
        seed=17,
        cue_size=2,
        trials=3,
    )
    assert replay.cue_a_ids == result.cue_a_ids
    assert replay.cue_b_ids == result.cue_b_ids
    assert np.isclose(replay.trained_after, result.trained_after)


def test_target_selection_counts_distinct_kc_inputs(tmp_path: Path) -> None:
    snapshot = association_snapshot(tmp_path / "snapshot", duplicate_bias=True)

    result = run_mb_association(
        snapshot,
        state_path=tmp_path / "memory.npz",
        seed=1,
        cue_size=1,
        trials=3,
    )

    assert result.target_mbon_id == 11
    assert result.target_connected_kcs == 3


def test_cli_writes_association_metrics_and_plastic_state(tmp_path: Path) -> None:
    snapshot = association_snapshot(tmp_path / "snapshot")
    output = tmp_path / "metrics.json"
    state = tmp_path / "memory.npz"

    result = runner.invoke(
        app,
        [
            "experiment",
            "mb-association",
            str(snapshot),
            "--seed",
            "17",
            "--cue-size",
            "2",
            "--trials",
            "3",
            "--state-output",
            str(state),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    metrics = json.loads(result.stdout)
    assert metrics["target_mbon_id"] == 10
    assert metrics["persistence_replay_exact"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["dataset_id"] == "association-fixture"
    assert state.is_file()
