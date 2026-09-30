"""Premotor control selection must be deterministic and fail closed."""

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.premotor_output_probe import _edge_record, _select_controls


def _edge(source: int, post: int, weight: float) -> dict:
    return {"source_id": source, "post_id": post, "position": source, "weight": weight}


def test_control_selection_uses_prior_exposure_and_deterministic_ties() -> None:
    route = [_edge(1, 10, -10.0)]
    candidates = {10: [
        _edge(2, 10, -10.0), _edge(3, 10, -5.0),
        _edge(4, 10, -5.0), _edge(5, 10, -10.0),
    ]}
    prior = [{1: 2, 2: 2, 3: 2, 4: 2, 5: 2} for _ in range(6)]

    singles, pairs, exposures = _select_controls(route, candidates, prior)

    assert [item["source_id"] for item in singles] == [2]
    assert [item["source_id"] for item in pairs] == [3, 4]
    assert exposures[0]["route"] == (20.0,) * 6
    assert exposures[0]["pair"] == (20.0,) * 6
    assert _select_controls(route, candidates, prior) == (singles, pairs, exposures)


def test_control_selection_rejects_incomplete_prior_panel() -> None:
    with pytest.raises(ValueError, match="six prior"):
        _select_controls([_edge(1, 10, -10.0)], {10: [_edge(2, 10, -10.0)]}, [])


def test_edge_record_rejects_absent_canonical_edge() -> None:
    outgoing = csr_array(
        (np.asarray([-10.0], dtype=np.float32), ([0], [1])), shape=(3, 3)
    )
    graph = EventConnectome(
        neuron_ids=np.asarray([1, 2, 3], dtype=np.uint64),
        cell_types=("a", "b", "c"),
        roles=("a", "b", "c"),
        transmitters=("gaba", "gaba", "gaba"),
        superclasses=("vnc_intrinsic", "vnc_motor", "vnc_motor"),
        outgoing=outgoing,
    )
    by_id = {1: 0, 2: 1, 3: 2}
    assert _edge_record(graph, by_id, 1, 2)["weight"] == -10.0
    with pytest.raises(ValueError, match="exactly once"):
        _edge_record(graph, by_id, 1, 3)
