"""Same-post VNC controls must be anatomically real and fail closed."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.dna02_same_post_probe import (
    PAIR_SOURCES,
    SINGLE_SOURCES,
    run_same_post_probe,
)
from flybrain.dng33_edge_probe import _graph_digest
from flybrain.graph import EventConnectome
from flybrain.odor_mbon_probe import _graph_digest as _odor_graph_digest


def _fixture(*, omit_last_control: bool = False) -> tuple[EventConnectome, dict, dict]:
    food, threat = 700001, 700002
    dna02 = (10360, 523769)
    targets = tuple(SINGLE_SOURCES)
    controls = set(SINGLE_SOURCES.values()) | {
        source for pair in PAIR_SOURCES.values() for source in pair
    }
    ids = tuple(sorted((food, threat, *dna02, *targets, *controls)))
    by_id = {value: i for i, value in enumerate(ids)}
    edges: dict[tuple[int, int], float] = {}
    for orn in (food, threat):
        for source in (*dna02, *controls):
            edges[orn, source] = 300.0
    for i, post in enumerate(targets):
        edges[dna02[0] if i < 8 else dna02[1], post] = 150.0
        for source in {SINGLE_SOURCES[post], *PAIR_SOURCES[post]}:
            if omit_last_control and post == targets[-1] and source == PAIR_SOURCES[post][-1]:
                continue
            edges[source, post] = 50.0
    outgoing = csr_array(
        (
            np.asarray(list(edges.values()), dtype=np.float32),
            (
                [by_id[pre] for pre, _ in edges],
                [by_id[post] for _, post in edges],
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
            "descending_neuron" if value in dna02 else
            "vnc_motor" if value in targets else
            "vnc_intrinsic" if value in controls else "sensory"
            for value in ids
        ),
        outgoing=outgoing,
    )
    target_edges = []
    for i, post in enumerate(targets):
        source = dna02[0] if i < 8 else dna02[1]
        start = int(outgoing.indptr[by_id[source]])
        stop = int(outgoing.indptr[by_id[source] + 1])
        matches = np.flatnonzero(outgoing.indices[start:stop] == by_id[post])
        assert matches.size == 1
        position = start + int(matches[0])
        target_edges.append({
            "source_id": source, "post_id": post, "position": position,
            "weight": float(outgoing.data[position]),
        })
    frozen = {
        "graph_digest": _graph_digest(graph),
        "source_ids": dna02,
        "target_ids": targets,
        "target_edges": target_edges,
    }
    odor = {
        "graph_digest": _odor_graph_digest(graph),
        "steps": 2_000,
        "source_ids": {"food": [food], "threat": [threat]},
    }
    return graph, frozen, odor


def test_same_post_probe_preserves_source_timing_and_graph() -> None:
    graph, frozen, odor = _fixture()
    before = graph.outgoing.data.copy()
    result = run_same_post_probe(graph, frozen, odor, steps=500, seeds=(7,))
    assert len(result["single_control_edges"]) == 17
    assert len(result["pair_control_edges"]) == 34
    assert len(result["seeds"]) == 2
    for row in result["seeds"]:
        assert row["gates"]["paired_events"]
        assert row["gates"]["source_timing_paired"]
        assert row["gates"]["other_motor_equal_all_conditions"]
        assert row["gates"]["no_source_quiet"]
        assert row["gates"]["replay_exact"]
    assert result["graph_unchanged"]
    assert np.array_equal(graph.outgoing.data, before)
    assert result["autonomous_behavior_claim_allowed"] is False


def test_same_post_probe_rejects_missing_control_edge() -> None:
    graph, frozen, odor = _fixture(omit_last_control=True)
    with pytest.raises(ValueError, match="same-post control edge"):
        run_same_post_probe(graph, frozen, odor, steps=10, seeds=(7,))
