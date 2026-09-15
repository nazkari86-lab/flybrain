import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.plasticity import (
    PlasticEdgeSet,
    PlasticityParameters,
    PlasticStateIdentity,
    load_plastic_state,
    save_plastic_state,
)


def edge_set() -> PlasticEdgeSet:
    return PlasticEdgeSet.create(
        pre_ids=np.array([1, 2, 3], dtype=np.uint64),
        post_ids=np.array([10, 10, 11], dtype=np.uint64),
        baseline_weights=np.array([5.0, 6.0, 7.0], dtype=np.float32),
    )


def parameters() -> PlasticityParameters:
    return PlasticityParameters(
        eligibility_tau_ms=1000.0,
        learning_rate=0.1,
        min_multiplier=0.25,
        max_multiplier=1.5,
    )


def identity() -> PlasticStateIdentity:
    return PlasticStateIdentity(
        dataset_id="fixture",
        source_manifest_sha256="a" * 64,
        snapshot_sha256="b" * 64,
    )


def test_dopamine_changes_only_recently_active_cue_target_edges() -> None:
    edges = edge_set()
    edges.update_eligibility(
        active_pre_ids=np.array([1], dtype=np.uint64),
        gated_post_ids=np.array([10], dtype=np.uint64),
        dt_ms=1.0,
        params=parameters(),
    )

    edges.apply_dopamine({10: 1.0}, parameters())

    np.testing.assert_allclose(edges.multipliers, np.array([0.9, 1.0, 1.0]))


def test_no_dopamine_causes_no_weight_change() -> None:
    edges = edge_set()
    edges.update_eligibility(
        active_pre_ids=np.array([1], dtype=np.uint64),
        gated_post_ids=np.array([10], dtype=np.uint64),
        dt_ms=1.0,
        params=parameters(),
    )

    edges.apply_dopamine({}, parameters())

    np.testing.assert_array_equal(edges.multipliers, np.ones(3, dtype=np.float32))


def test_eligibility_decays_and_multiplier_respects_lower_bound() -> None:
    edges = edge_set()
    params = parameters()
    edges.update_eligibility(
        active_pre_ids=np.array([1], dtype=np.uint64),
        gated_post_ids=np.array([10], dtype=np.uint64),
        dt_ms=1.0,
        params=params,
    )
    edges.update_eligibility(
        active_pre_ids=np.array([], dtype=np.uint64),
        gated_post_ids=np.array([], dtype=np.uint64),
        dt_ms=1000.0,
        params=params,
    )

    assert np.isclose(edges.eligibility[0], np.exp(-1.0), rtol=1e-6)
    for _ in range(100):
        edges.apply_dopamine({10: 1.0}, params)
    assert edges.multipliers[0] == params.min_multiplier


def test_signed_dopamine_respects_upper_bound() -> None:
    edges = edge_set()
    params = parameters()
    edges.update_eligibility(
        active_pre_ids=np.array([1], dtype=np.uint64),
        gated_post_ids=np.array([10], dtype=np.uint64),
        dt_ms=0.0,
        params=params,
    )

    for _ in range(100):
        edges.apply_dopamine({10: -1.0}, params)

    assert edges.multipliers[0] == params.max_multiplier


def test_plastic_state_round_trip_preserves_observable_weights(tmp_path: Path) -> None:
    edges = edge_set()
    params = parameters()
    edges.update_eligibility(
        active_pre_ids=np.array([1, 3], dtype=np.uint64),
        gated_post_ids=np.array([10, 11], dtype=np.uint64),
        dt_ms=1.0,
        params=params,
    )
    edges.apply_dopamine({10: 0.5, 11: -0.25}, params)
    path = tmp_path / "memory.npz"

    save_plastic_state(path, edges, params, identity())
    restored, restored_params, restored_identity = load_plastic_state(path)

    assert restored_params == params
    assert restored_identity == identity()
    np.testing.assert_array_equal(restored.pre_ids, edges.pre_ids)
    np.testing.assert_array_equal(restored.post_ids, edges.post_ids)
    np.testing.assert_array_equal(restored.effective_weights, edges.effective_weights)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligibility_tau_ms", float("nan")),
        ("learning_rate", float("inf")),
        ("min_multiplier", float("nan")),
        ("max_multiplier", float("inf")),
    ],
)
def test_rejects_non_finite_plasticity_parameters(field: str, value: float) -> None:
    values = parameters().__dict__ | {field: value}

    with pytest.raises(ValueError, match="finite"):
        PlasticityParameters(**values).validate()


def test_rejects_non_finite_baseline_and_dopamine() -> None:
    with pytest.raises(ValueError, match="finite"):
        PlasticEdgeSet.create(
            pre_ids=np.array([1], dtype=np.uint64),
            post_ids=np.array([10], dtype=np.uint64),
            baseline_weights=np.array([np.nan], dtype=np.float32),
        )

    edges = edge_set()
    with pytest.raises(ValueError, match="finite"):
        edges.apply_dopamine({10: float("nan")}, parameters())


def test_rejects_malformed_persisted_arrays(tmp_path: Path) -> None:
    path = tmp_path / "malformed.npz"
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            pre_ids=np.array([[1]], dtype=np.uint64),
            post_ids=np.array([10], dtype=np.uint64),
            baseline_weights=np.array([5.0], dtype=np.float32),
            multipliers=np.array([np.nan], dtype=np.float32),
            eligibility=np.array([-1.0], dtype=np.float32),
            parameters=np.array(json.dumps(parameters().__dict__)),
            identity=np.array(json.dumps(identity().__dict__)),
        )

    with pytest.raises(ValueError, match=r"one-dimensional|finite|non-negative"):
        load_plastic_state(path)
