"""BANC route audit keeps provenance, side and segment boundaries explicit."""

import pyarrow as pa
import pyarrow.feather as feather

from flybrain.banc_route_audit import (
    _classify_edges,
    _relevant_edges,
    _segment,
    _selected_cells,
)


def _cell(
    root: str, kind: str, side: str, segment: str | None = None,
    *, muscle: str | None = None, region: str | None = None,
) -> dict[str, str | None]:
    return {
        "banc_888_id": root,
        "cell_type": kind,
        "side": side,
        "neuromere": segment,
        "root_region": region,
        "proofread": "TRUE",
        "flow": "efferent" if muscle else "intrinsic",
        "peripheral_target_type": muscle,
    }


def _fixture() -> tuple[list[dict[str, str | None]], dict[tuple[str, str, str], str]]:
    base = 9_007_199_254_740_993
    rows = []
    ids = {}
    for side in ("left", "right"):
        for kind in ("DNa02", "DNg13"):
            root = str(base + len(rows))
            rows.append(_cell(root, kind, side))
            ids[kind, side, "source"] = root
        for segment in ("T1", "T2", "T3"):
            for kind, muscle in (
                ("IN19A003", "sternal_posterior_rotator_muscle"),
                ("IN08A006", "sternal_anterior_rotator_muscle"),
            ):
                root = str(base + len(rows))
                rows.append(_cell(root, kind, side, segment))
                ids[kind, side, segment] = root
                motor = str(base + len(rows))
                missing = side == "left" and segment == "T3" and kind == "IN19A003"
                region = "MANC_vnc_LNp_T3_L" if missing else None
                rows.append(_cell(
                    motor, "motor", side, None if missing else segment,
                    muscle=muscle, region=region,
                ))
                ids[muscle, side, segment] = motor
    return rows, ids


def test_selection_requires_proofreading_and_preserves_large_string_ids() -> None:
    rows, ids = _fixture()
    extra = _cell("9007199254749999", "IN19A003", "left", "T1")
    extra["proofread"] = "FALSE"
    selected = _selected_cells([*rows, extra])

    assert len(selected) == len(rows)
    assert ids["DNa02", "left", "source"] in {
        row["banc_888_id"] for row in selected
    }
    assert all(isinstance(row["banc_888_id"], str) for row in selected)


def test_inferred_segment_is_separate_from_primary_and_checks_side() -> None:
    rows, ids = _fixture()
    target = next(
        row for row in rows
        if row["banc_888_id"] == ids[
            "sternal_posterior_rotator_muscle", "left", "T3"
        ]
    )
    assert _segment(target, infer_root_region=False) is None
    assert _segment(target, infer_root_region=True) == "T3"
    assert _segment({**target, "side": "right"}, infer_root_region=True) is None


def test_edge_threshold_and_same_side_segment_gates() -> None:
    rows, ids = _fixture()
    def edge(pre: str, post: str, count: int) -> dict[str, str | int]:
        return {"pre": pre, "post": post, "count": count}

    observed = [
        edge(ids["DNa02", "left", "source"], ids["IN19A003", "left", "T1"], 5),
        edge(ids["DNa02", "left", "source"], ids["IN08A006", "left", "T1"], 4),
        edge(ids["DNg13", "left", "source"], ids["IN19A003", "left", "T1"], 7),
        edge(ids["DNa02", "right", "source"], ids["IN19A003", "left", "T1"], 9),
        edge(ids["IN19A003", "left", "T1"], ids[
            "sternal_posterior_rotator_muscle", "left", "T1"
        ], 6),
        edge(ids["IN19A003", "left", "T3"], ids[
            "sternal_posterior_rotator_muscle", "left", "T3"
        ], 8),
        edge(ids["IN08A006", "left", "T1"], ids[
            "sternal_anterior_rotator_muscle", "left", "T1"
        ], 4),
        edge(ids["IN19A003", "left", "T1"], ids[
            "sternal_posterior_rotator_muscle", "right", "T1"
        ], 20),
    ]
    result = _classify_edges(rows, observed)

    assert result["dna02_to_same_side_relays"]["observed_edges"] == 2
    assert result["dna02_to_same_side_relays"]["passing_edges"] == 1
    assert result["dng13_to_same_side_relays"]["passing_edges"] == 1
    assert result["relay_to_muscle_primary"]["passing_edges"] == 1
    assert result["relay_to_muscle_with_root_region_inference"]["passing_edges"] == 2
    assert result["relay_to_muscle_primary"]["eligible_pairs_or_segments"] == 11


def test_arrow_reader_filters_without_float_coercion(tmp_path) -> None:
    path = tmp_path / "edges.feather"
    large = "9007199254740993"
    table = pa.table({
        "pre": [large, "9007199254740994", large],
        "post": ["9007199254740995", "9007199254740995", "9007199254740996"],
        "count": pa.array([5, 7, 3], type=pa.int32()),
    })
    feather.write_feather(table, path)

    rows = list(_relevant_edges(path, {large}, {"9007199254740995"}))

    assert rows == [{"pre": large, "post": "9007199254740995", "count": 5}]
