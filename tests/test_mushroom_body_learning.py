import numpy as np
import pytest

from flybrain.mushroom_body_learning import (
    MushroomBodyLearning,
    MushroomBodyLearningParameters,
    derive_approach_mbon_ids,
)
from flybrain.plastic_overlay import PlasticWeightOverlay


def test_dan_and_eligibility_are_both_required_for_local_weight_change() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(),
    )

    learning.step(
        active_kc_ids=np.array([10], dtype=np.uint64),
        active_mbon_ids=np.array([20], dtype=np.uint64),
        routed_dan_ids=np.array([], dtype=np.uint64),
        dt_ms=10.0,
    )
    assert overlay.multipliers.tolist() == [1.0]

    dan_only_overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    dan_only = MushroomBodyLearning(
        overlay=dan_only_overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(),
    )
    dan_only.step(
        active_kc_ids=np.array([], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([30], dtype=np.uint64),
        dt_ms=10.0,
    )
    assert dan_only_overlay.multipliers.tolist() == [1.0]

    learning.step(
        active_kc_ids=np.array([], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([30], dtype=np.uint64),
        dt_ms=10.0,
    )
    assert overlay.multipliers[0] < 1.0


def test_valence_compartments_switch_between_potentiation_and_depression() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([0, 1], dtype=np.int64), canonical_edge_count=2
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10, 10], dtype=np.uint64),
        edge_post_ids=np.array([20, 21], dtype=np.uint64),
        dan_ids=np.array([30, 30], dtype=np.uint64),
        dan_post_ids=np.array([20, 21], dtype=np.uint64),
        appetitive_dan_ids=np.array([30], dtype=np.uint64),
        aversive_dan_ids=np.array([31], dtype=np.uint64),
        approach_mbon_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(
            plasticity_mode="valence_compartmental",
            maximum_multiplier=2.0,
        ),
    )

    learning.step(
        active_kc_ids=np.array([10], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([30], dtype=np.uint64),
        dt_ms=0.0,
    )

    assert overlay.multipliers[0] > 1.0
    assert overlay.multipliers[1] < 1.0


def test_aversive_dan_majority_derives_unambiguous_approach_compartments() -> None:
    approach = derive_approach_mbon_ids(
        dan_ids=np.array([30, 30, 31, 31, 31], dtype=np.uint64),
        dan_post_ids=np.array([20, 20, 21, 22, 22], dtype=np.uint64),
        appetitive_dan_ids=np.array([30], dtype=np.uint64),
        aversive_dan_ids=np.array([31], dtype=np.uint64),
    )

    assert approach.tolist() == [21, 22]


def test_unrouted_dan_cannot_change_an_eligible_edge() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(),
    )

    learning.step(
        active_kc_ids=np.array([10], dtype=np.uint64),
        active_mbon_ids=np.array([20], dtype=np.uint64),
        routed_dan_ids=np.array([31], dtype=np.uint64),
        dt_ms=10.0,
    )

    assert overlay.multipliers.tolist() == [1.0]


def test_presynaptic_kc_eligibility_does_not_require_an_mbon_spike() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(),
    )

    learning.step(
        active_kc_ids=np.array([10], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([30], dtype=np.uint64),
        dt_ms=10.0,
    )

    assert overlay.multipliers[0] < 1.0


def test_dan_trace_can_gate_later_kc_eligibility_in_the_same_compartment() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(dopamine_trace_tau_ms=100.0),
    )

    learning.step(
        active_kc_ids=np.array([], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([30], dtype=np.uint64),
        dt_ms=0.0,
    )
    learning.step(
        active_kc_ids=np.array([10], dtype=np.uint64),
        active_mbon_ids=np.array([], dtype=np.uint64),
        routed_dan_ids=np.array([], dtype=np.uint64),
        dt_ms=30.0,
    )

    assert overlay.multipliers[0] < 1.0


def test_repeated_routed_updates_respect_the_declared_weight_floor() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.array([3], dtype=np.int64), canonical_edge_count=4
    )
    parameters = MushroomBodyLearningParameters(
        eligibility_tau_ms=1_000_000.0,
        learning_rate=0.8,
        minimum_multiplier=0.2,
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10], dtype=np.uint64),
        edge_post_ids=np.array([20], dtype=np.uint64),
        dan_ids=np.array([30], dtype=np.uint64),
        dan_post_ids=np.array([20], dtype=np.uint64),
        parameters=parameters,
    )

    for _ in range(5):
        learning.step(
            active_kc_ids=np.array([10], dtype=np.uint64),
            active_mbon_ids=np.array([20], dtype=np.uint64),
            routed_dan_ids=np.array([30], dtype=np.uint64),
            dt_ms=1.0,
        )

    assert overlay.multipliers.tolist() == pytest.approx([0.2])


def test_repeated_unsorted_mbon_endpoints_share_only_their_routed_dan_trace() -> None:
    overlay = PlasticWeightOverlay.create(
        edge_indices=np.arange(5, dtype=np.int64), canonical_edge_count=5
    )
    learning = MushroomBodyLearning(
        overlay=overlay,
        edge_pre_ids=np.array([10, 11, 12, 13, 14], dtype=np.uint64),
        edge_post_ids=np.array([21, 20, 21, 22, 20], dtype=np.uint64),
        dan_ids=np.array([30, 31, 30, 32], dtype=np.uint64),
        dan_post_ids=np.array([20, 21, 21, 20], dtype=np.uint64),
        parameters=MushroomBodyLearningParameters(),
    )
    learning.dopamine_trace[:] = [1.0, 4.0, 2.0, 8.0]
    np.testing.assert_array_equal(
        learning._route_trace(np.array([30], dtype=np.uint64)),
        np.array([2.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        learning._route_trace(learning.dan_ids),
        np.array([6.0, 9.0, 6.0, 0.0, 9.0], dtype=np.float32),
    )
