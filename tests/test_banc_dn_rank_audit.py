"""The cross-animal ranking must respect cohort, direction, side and threshold."""

import pytest

from flybrain.banc_dn_rank_audit import _cohort, _rank


def _rows() -> tuple[list[dict[str, str]], dict[tuple[str, str, str], str]]:
    base = 9_007_199_254_740_993
    rows = []
    ids = {}
    for kind in ("DNa02", "DNge023", "DNg13"):
        for side in ("left", "right"):
            root = str(base + len(rows))
            rows.append({"banc_888_id": root, "cell_type": kind, "side": side,
                         "neuromere": "", "proofread": "TRUE",
                         "super_class": "descending"})
            ids[kind, side, "source"] = root
    for kind in ("IN19A003", "IN08A006"):
        for side in ("left", "right"):
            for segment in ("T1", "T2", "T3"):
                root = str(base + len(rows))
                rows.append({"banc_888_id": root, "cell_type": kind, "side": side,
                             "neuromere": segment, "proofread": "TRUE",
                             "super_class": "intrinsic"})
                ids[kind, side, segment] = root
    return rows, ids


def test_cohort_requires_exact_bilateral_pair_and_relay_inventory() -> None:
    rows, ids = _rows()
    sources, relays = _cohort(rows)
    assert len(sources) == 6
    assert len(relays) == 12
    assert ids["DNa02", "left", "source"] in {
        row["banc_888_id"] for row in sources
    }
    extra = {**sources[-1], "banc_888_id": "9007199254741999"}
    selected, _ = _cohort([*rows, extra])
    assert all(row["cell_type"] != "DNge023" for row in selected)
    with pytest.raises(ValueError, match="inventory"):
        _cohort(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _cohort([{**row, "banc_888_id": ids["DNa02", "left", "source"]}
                 if row["cell_type"] == "DNg13" and row["side"] == "left"
                 else row for row in rows])


def test_rank_is_same_side_thresholded_coverage_then_weight() -> None:
    rows, ids = _rows()
    sources, relays = _cohort(rows)
    edges = [
        {"pre": ids["DNa02", "left", "source"],
         "post": ids["IN19A003", "left", "T1"], "count": 5},
        {"pre": ids["DNa02", "right", "source"],
         "post": ids["IN08A006", "right", "T2"], "count": 9},
        {"pre": ids["DNa02", "left", "source"],
         "post": ids["IN08A006", "left", "T2"], "count": 4},
        {"pre": ids["DNa02", "right", "source"],
         "post": ids["IN19A003", "left", "T2"], "count": 100},
        {"pre": ids["DNge023", "left", "source"],
         "post": ids["IN19A003", "left", "T1"], "count": 6},
        {"pre": ids["DNge023", "right", "source"],
         "post": ids["IN08A006", "right", "T2"], "count": 6},
    ]
    result = _rank(sources, relays, edges)
    assert [(row["cell_type"], row["coverage"], row["synapses"])
            for row in result["ranking"]] == [
                ("DNa02", 2, 14), ("DNge023", 2, 12), ("DNg13", 0, 0),
            ]
    assert result["dna02"]["rank"] == 1
    assert len(result["dna02_edges"]) == 2
    assert result["full_coverage_types"] == []


def test_rank_rejects_duplicate_or_invalid_edges() -> None:
    rows, ids = _rows()
    sources, relays = _cohort(rows)
    edge = {"pre": ids["DNa02", "left", "source"],
            "post": ids["IN19A003", "left", "T1"], "count": 5}
    with pytest.raises(ValueError, match="duplicate directed"):
        _rank(sources, relays, [edge, edge])
    with pytest.raises(ValueError, match="positive integer"):
        _rank(sources, relays, [{**edge, "count": 0}])
    with pytest.raises(ValueError, match="outside selected"):
        _rank(sources, relays, [{**edge, "pre": "9007199254742999"}])
