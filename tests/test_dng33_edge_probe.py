"""The fixed-source assay must isolate the declared DNg33 edges, not cell IDs."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_array

from flybrain.dng33_edge_probe import (
    CONTROL_IDS,
    SOURCE_IDS,
    TARGET_IDS,
    resolve_frozen_edges,
    run_dng33_edge_probe,
)
from flybrain.graph import EventConnectome


def _graph() -> EventConnectome:
    target_weights = (
        (14, 34, 16, 15, 42, 25, 29, 49),
        (7, 34, 21, 9, 39, 13, 29, 28),
    )
    control_weights = (
        (13, 37, 16, 13, 37, 22, 28, 40),
        (6, 33, 22, 12, 36, 16, 24, 24),
    )
    ids = tuple(sorted(SOURCE_IDS + TARGET_IDS + CONTROL_IDS + (999999,)))
    by_id = {value: index for index, value in enumerate(ids)}
    pre: list[int] = []
    post: list[int] = []
    weights: list[float] = []
    for source_index, source in enumerate(SOURCE_IDS):
        for posts, values in (
            (TARGET_IDS, target_weights[source_index]),
            (CONTROL_IDS, control_weights[source_index]),
        ):
            for target, weight in zip(posts, values, strict=True):
                pre.append(by_id[source])
                post.append(by_id[target])
                weights.append(float(weight))
    outgoing = csr_array((weights, (pre, post)), shape=(len(ids), len(ids)))
    outgoing.sort_indices()
    superclasses = tuple(
        "vnc_motor" if value in TARGET_IDS or value == 999999
        else "vnc_intrinsic" if value in CONTROL_IDS
        else "descending_neuron"
        for value in ids
    )
    return EventConnectome(
        neuron_ids=np.asarray(ids, dtype=np.uint64),
        cell_types=tuple("fixture" for _ in ids),
        roles=tuple("motor" if value in TARGET_IDS else "other" for value in ids),
        transmitters=tuple("acetylcholine" for _ in ids),
        superclasses=superclasses,
        outgoing=outgoing,
    )


def test_direct_edge_removal_blocks_target_spikes_without_changing_source() -> None:
    """Catches zeroing the wrong CSR rows or substituting biological IDs for indices."""
    graph = _graph()
    original_data = graph.outgoing.data.copy()
    result = run_dng33_edge_probe(graph, steps=2_000, seeds=(0,))

    seed = result.seeds[0]
    normal = seed.conditions["intact"]
    direct = seed.conditions["direct_edges_zero"]
    matched = seed.conditions["matched_edges_zero"]
    assert len(result.target_edges) == 16
    assert sum(edge.weight for edge in result.target_edges) == 404
    assert sum(edge.weight for edge in result.control_edges) == 379
    assert sum(normal.target_counts.values()) > 0
    assert sum(direct.target_counts.values()) == 0
    assert matched.target_counts == normal.target_counts
    assert normal.source_spike_digest == direct.source_spike_digest
    assert normal.source_spike_digest == matched.source_spike_digest
    assert normal.source_event_digest == direct.source_event_digest
    assert result.graph_unchanged
    assert seed.primary_gate_passed
    assert np.array_equal(graph.outgoing.data, original_data)
    assert result.autonomous_behavior_claim_allowed is False


@pytest.mark.parametrize("replacement", [0.0, -14.0])
def test_missing_or_nonpositive_declared_edge_fails_closed(replacement: float) -> None:
    """Catches accepting absent or sign-reversed anatomy as a valid lesion."""
    graph = _graph()
    edge = resolve_frozen_edges(graph, SOURCE_IDS, TARGET_IDS)[0]
    modified = graph.outgoing.copy()
    modified.data[edge.position] = replacement
    with pytest.raises(ValueError, match="positive"):
        run_dng33_edge_probe(replace(graph, outgoing=modified), steps=20, seeds=(0,))


def test_module_cli_writes_provenance_and_frozen_edge_panel(tmp_path: Path) -> None:
    """Catches a CLI that omits the causal conditions or immutable provenance."""
    graph = _graph()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.joinpath("metadata.json").write_text('{"dataset_id":"fixture"}', encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "neuron_id": graph.neuron_ids,
                "cell_type": graph.cell_types,
                "role": graph.roles,
                "transmitter": graph.transmitters,
                "superclass": graph.superclasses,
            }
        ),
        snapshot / "neurons.parquet",
    )
    pre: list[int] = []
    post: list[int] = []
    weight: list[int] = []
    for pre_index in range(graph.neuron_count):
        for edge_index in range(
            graph.outgoing.indptr[pre_index], graph.outgoing.indptr[pre_index + 1]
        ):
            pre.append(int(graph.neuron_ids[pre_index]))
            post.append(int(graph.neuron_ids[graph.outgoing.indices[edge_index]]))
            weight.append(int(graph.outgoing.data[edge_index]))
    pq.write_table(
        pa.table(
            {
                "pre_id": pre,
                "post_id": post,
                "synapse_count": weight,
                "sign": [1] * len(pre),
            }
        ),
        snapshot / "edges.parquet",
    )
    output = tmp_path / "probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "flybrain.dng33_edge_probe",
            str(snapshot),
            "--output",
            str(output),
            "--steps",
            "100",
            "--seeds",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["snapshot_content_sha256"]) == 64
    assert payload["software_revision"]
    assert payload["graph_neurons"] == 19
    assert len(payload["target_edges"]) == 16
    assert len(payload["control_edges"]) == 16
    assert set(payload["seeds"][0]["conditions"]) == {
        "intact",
        "direct_edges_zero",
        "matched_edges_zero",
        "no_source",
        "replay",
    }
    assert payload["autonomous_behavior_claim_allowed"] is False
    assert subprocess.run(
        [sys.executable, "-m", "flybrain.dng33_edge_probe", str(snapshot), "--output", str(output)],
        capture_output=True,
        check=False,
    ).returncode != 0
    for invalid_seeds in (("-1",), ("0", "0")):
        invalid = subprocess.run(
            [
                sys.executable,
                "-m",
                "flybrain.dng33_edge_probe",
                str(snapshot),
                "--output",
                str(tmp_path / "invalid.json"),
                "--seeds",
                *invalid_seeds,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert invalid.returncode != 0
        assert "usage:" in invalid.stderr
        assert "Traceback" not in invalid.stderr
    for invalid_args in (
        (str(snapshot),),
        (str(tmp_path / "missing-snapshot"), "--output", str(tmp_path / "invalid.json")),
        (str(snapshot), "--output", str(tmp_path / "invalid.json"), "--steps", "0"),
    ):
        invalid = subprocess.run(
            [sys.executable, "-m", "flybrain.dng33_edge_probe", *invalid_args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert invalid.returncode != 0
        assert "usage:" in invalid.stderr
