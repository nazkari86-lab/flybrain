"""Afferent-isolated DNa02 edge assay must hold source spikes fixed."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.dna02_afferent_probe import run_dna02_afferent_probe
from flybrain.dng33_edge_probe import _condition, _graph_digest
from flybrain.graph import EventConnectome
from flybrain.shiu import ShiuParameters, poisson_voltage_events


def _fixture() -> tuple[EventConnectome, dict]:
    sources = (10360, 523769)
    targets = tuple(900100 + i for i in range(17))
    controls = tuple(800100 + i for i in range(17))
    ids = tuple(sorted(sources + targets + controls))
    by_id = {value: index for index, value in enumerate(ids)}
    pre: list[int] = []
    post: list[int] = []
    weight: list[float] = []
    for i, (target, control) in enumerate(zip(targets, controls, strict=True)):
        source = sources[0] if i < 7 else sources[1]
        for left, right, value in (
            (source, target, 35.0),
            (source, control, 35.0),
            (control, source, 100.0),
        ):
            pre.append(by_id[left])
            post.append(by_id[right])
            weight.append(value)
    outgoing = csr_array((weight, (pre, post)), shape=(len(ids), len(ids)))
    outgoing.sort_indices()
    graph = EventConnectome(
        neuron_ids=np.asarray(ids, dtype=np.uint64),
        cell_types=tuple("fixture" for _ in ids),
        roles=tuple("motor" if value in targets else "other" for value in ids),
        transmitters=tuple("acetylcholine" for _ in ids),
        superclasses=tuple(
            "descending_neuron" if value in sources else
            "vnc_motor" if value in targets else "vnc_intrinsic"
            for value in ids
        ),
        outgoing=outgoing,
    )

    def edges(posts: tuple[int, ...]) -> list[dict]:
        result = []
        for i, value in enumerate(posts):
            source = sources[0] if i < 7 else sources[1]
            start = int(outgoing.indptr[by_id[source]])
            stop = int(outgoing.indptr[by_id[source] + 1])
            matches = np.flatnonzero(outgoing.indices[start:stop] == by_id[value])
            assert len(matches) == 1
            position = start + int(matches[0])
            result.append({
                "source_id": source,
                "post_id": value,
                "position": position,
                "weight": float(outgoing.data[position]),
            })
        return result

    return graph, {
        "source_ids": list(sources),
        "target_ids": list(targets),
        "control_ids": list(controls),
        "target_edges": edges(targets),
        "control_edges": edges(controls),
        "graph_digest": _graph_digest(graph),
    }


def test_common_afferent_overlay_pairs_source_spikes_and_selects_targets() -> None:
    graph, frozen = _fixture()
    before = graph.outgoing.data.copy()
    by_id = {int(value): i for i, value in enumerate(graph.neuron_ids)}
    source_ids = tuple(frozen["source_ids"])
    target_ids = tuple(frozen["target_ids"])
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)
    events = poisson_voltage_events(
        np.asarray([by_id[value] for value in source_ids], dtype=np.int64),
        steps=1_000, params=parameters, seed=2,
    )
    unisolated_args = dict(
        source_ids=source_ids, target_ids=target_ids, other_motor_ids=set(),
        events=events, steps=1_000, seed=2, parameters=parameters,
    )
    unisolated = _condition(graph, name="intact", edge_positions=None, **unisolated_args)
    unisolated_control = _condition(
        graph, name="control_zero",
        edge_positions=np.asarray(
            sorted(edge["position"] for edge in frozen["control_edges"]),
            dtype=np.int64,
        ),
        **unisolated_args,
    )
    assert unisolated.source_spike_digest != unisolated_control.source_spike_digest
    result = run_dna02_afferent_probe(
        graph, frozen, seeds=(2,), steps=1_000, expected_afferent_count=17,
    )
    row = result["seeds"][0]
    conditions = row["conditions"]
    intact = conditions["intact"]
    direct = conditions["direct_edges_zero"]
    matched = conditions["matched_edges_zero"]
    assert result["primary_gate_passed"]
    assert result["source_incoming_counts"] == {"10360": 7, "523769": 10}
    assert len(result["source_afferent_zero_positions"]) == 17
    assert intact["source_spikes"] > 0
    assert intact["source_spike_digest"] == direct["source_spike_digest"]
    assert intact["source_spike_digest"] == matched["source_spike_digest"]
    assert sum(intact["target_counts"].values()) > 0
    assert sum(direct["target_counts"].values()) == 0
    assert intact["target_counts"] == matched["target_counts"]
    assert sum(conditions["no_source"]["target_counts"].values()) == 0
    assert intact["trace_digest"] == conditions["replay"]["trace_digest"]
    assert result["graph_unchanged"]
    assert np.array_equal(graph.outgoing.data, before)
    assert result["autonomous_behavior_claim_allowed"] is False


def test_rejects_frozen_edge_drift_and_afferent_count_drift() -> None:
    graph, frozen = _fixture()
    modified = deepcopy(frozen)
    modified["target_edges"][0]["weight"] += 1
    with pytest.raises(ValueError, match="frozen edge"):
        run_dna02_afferent_probe(graph, modified, seeds=(2,), steps=10)
    with pytest.raises(ValueError, match="afferent"):
        run_dna02_afferent_probe(
            graph, frozen, seeds=(2,), steps=10, expected_afferent_count=18,
        )


def test_rejects_duplicate_seeds_before_running() -> None:
    graph, frozen = _fixture()
    with pytest.raises(ValueError, match="seeds"):
        run_dna02_afferent_probe(graph, frozen, seeds=(2, 2), steps=10)


def test_malformed_frozen_ids_fail_with_value_error() -> None:
    graph, frozen = _fixture()
    modified = deepcopy(frozen)
    modified["target_ids"] = None
    with pytest.raises(ValueError, match="frozen target and control IDs"):
        run_dna02_afferent_probe(graph, modified, seeds=(2,), steps=10)
