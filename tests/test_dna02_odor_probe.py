"""Registered-ORN DNa02 assay retains its natural afferents and fails closed."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.dna02_odor_probe import run_dna02_odor_probe
from flybrain.dng33_edge_probe import _graph_digest
from flybrain.graph import EventConnectome
from flybrain.odor_mbon_probe import _graph_digest as _odor_graph_digest


def _fixture() -> tuple[EventConnectome, dict, dict]:
    food, threat = 700001, 700002
    sources = (10360, 523769)
    targets = tuple(900100 + i for i in range(17))
    controls = tuple(800100 + i for i in range(17))
    ids = tuple(sorted((food, threat, *sources, *targets, *controls)))
    by_id = {value: i for i, value in enumerate(ids)}
    edges: list[tuple[int, int, float]] = []
    for receptor in (food, threat):
        for source in sources:
            edges.append((receptor, source, 300.0))
    for i, (target, control) in enumerate(zip(targets, controls, strict=True)):
        source = sources[0] if i < 7 else sources[1]
        edges.extend(((source, target, 150.0), (source, control, 150.0)))
    outgoing = csr_array(
        (
            np.asarray([weight for _, _, weight in edges], dtype=np.float32),
            (
                [by_id[pre] for pre, _, _ in edges],
                [by_id[post] for _, post, _ in edges],
            ),
        ),
        shape=(len(ids), len(ids)),
    )
    outgoing.sort_indices()
    graph = EventConnectome(
        neuron_ids=np.asarray(ids, dtype=np.uint64),
        cell_types=tuple(
            "ORN_DM1" if value == food else "ORN_DA2" if value == threat else "fixture"
            for value in ids
        ),
        roles=tuple("fixture" for _ in ids),
        transmitters=tuple("acetylcholine" for _ in ids),
        superclasses=tuple(
            "descending_neuron" if value in sources else
            "vnc_motor" if value in targets else
            "vnc_intrinsic" if value in controls else "sensory"
            for value in ids
        ),
        outgoing=outgoing,
    )

    def frozen_edges(posts: tuple[int, ...]) -> list[dict]:
        result = []
        for i, post in enumerate(posts):
            source = sources[0] if i < 7 else sources[1]
            start = int(outgoing.indptr[by_id[source]])
            stop = int(outgoing.indptr[by_id[source] + 1])
            matches = np.flatnonzero(outgoing.indices[start:stop] == by_id[post])
            assert matches.size == 1
            position = start + int(matches[0])
            result.append({
                "source_id": source, "post_id": post, "position": position,
                "weight": float(outgoing.data[position]),
            })
        return result

    digest = _graph_digest(graph)
    frozen = {
        "source_ids": sources,
        "target_ids": targets,
        "control_ids": controls,
        "target_edges": frozen_edges(targets),
        "control_edges": frozen_edges(controls),
        "graph_digest": digest,
    }
    odor = {
        "source_ids": {"food": [food], "threat": [threat]},
        "steps": 2_000,
        "graph_digest": _odor_graph_digest(graph),
    }
    return graph, frozen, odor


def test_registered_orn_assay_replays_and_preserves_source_timing() -> None:
    graph, frozen, odor = _fixture()
    before = graph.outgoing.data.copy()
    result = run_dna02_odor_probe(graph, frozen, odor, steps=1_000, seeds=(7,))
    assert result["narrow_gate_passed"]
    assert len(result["seeds"]) == 2
    for row in result["seeds"]:
        intact = row["conditions"]["intact"]
        direct = row["conditions"]["direct_edges_zero"]
        assert intact["source_voltage_events"] > 0
        assert intact["source_spike_digest"] == direct["source_spike_digest"]
        assert sum(direct["target_counts"].values()) < sum(intact["target_counts"].values())
        assert row["gates"]["replay_exact"]
    assert np.array_equal(graph.outgoing.data, before)
    assert result["autonomous_behavior_claim_allowed"] is False


def test_registered_orn_assay_rejects_source_and_edge_drift() -> None:
    graph, frozen, odor = _fixture()
    with pytest.raises(ValueError, match="unique"):
        run_dna02_odor_probe(graph, frozen, odor, steps=10, seeds=(7, 7))
    bad_odor = deepcopy(odor)
    bad_odor["source_ids"]["food"] = [10360]
    with pytest.raises(ValueError, match="annotated ORN"):
        run_dna02_odor_probe(graph, frozen, bad_odor, steps=10, seeds=(7,))
    bad_edges = deepcopy(frozen)
    bad_edges["target_edges"][0]["weight"] += 1
    with pytest.raises(ValueError, match="frozen edge"):
        run_dna02_odor_probe(graph, bad_edges, odor, steps=10, seeds=(7,))
