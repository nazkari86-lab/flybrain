import numpy as np
import pytest

from flybrain.mushroom_body_learning import MushroomBodyLearning, MushroomBodyLearningParameters
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
