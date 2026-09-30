"""Locked multi-input control selection never reads outcome panels."""

from collections import Counter

import pytest

from flybrain.premotor_multi_control_probe import _select_for_post


def _edge(source: int, weight: float) -> dict[str, int | float]:
    return {"source_id": source, "post_id": 10, "position": source, "weight": weight}


def test_four_active_inputs_can_match_one_strong_route_edge() -> None:
    route = _edge(1, -10.0)
    controls = [_edge(source, -2.5) for source in (5, 4, 3, 2)]
    counts = [Counter({source: 2 for source in range(1, 6)}) for _ in range(6)]

    chosen, route_vector, control_vector = _select_for_post(route, controls, counts)

    assert [edge["source_id"] for edge in chosen] == [2, 3, 4, 5]
    assert route_vector == (20.0,) * 6
    assert control_vector == route_vector


def test_exact_ties_use_weight_then_cardinality_then_source_id() -> None:
    route = _edge(1, -10.0)
    controls = [_edge(4, -5.0), _edge(3, -10.0), _edge(2, -10.0)]
    counts = [Counter({1: 2, 2: 2, 3: 2, 4: 4}) for _ in range(6)]

    chosen, _, _ = _select_for_post(route, controls, counts)

    assert [edge["source_id"] for edge in chosen] == [2]


def test_selection_rejects_incomplete_prior_panels_and_duplicate_sources() -> None:
    route = _edge(1, -10.0)
    with pytest.raises(ValueError, match="six prior"):
        _select_for_post(route, [_edge(2, -5.0)], [Counter({1: 2, 2: 2})])
    with pytest.raises(ValueError, match="duplicate"):
        _select_for_post(route, [_edge(2, -5.0), _edge(2, -4.0)], [Counter()] * 6)
