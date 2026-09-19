import numpy as np
import pytest

from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_controls import CONDITIONS, ControlCondition, build_condition
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay


def binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1, 3, 5], dtype=np.int64), canonical_edge_count=6
        ),
        pre_ids=np.array([10, 11, 12], dtype=np.uint64),
        post_ids=np.array([20, 21, 22], dtype=np.uint64),
    )


def learning() -> AssociativeCalibrationConfig:
    return AssociativeCalibrationConfig(
        cue_ids=(10, 11, 12),
        mbon_ids=(20, 21, 22),
        dan_to_mbon_pairs=((30, 20), (31, 21)),
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_conditions_are_independent_and_explicit(condition: ControlCondition) -> None:
    original = binding()
    result = build_condition(condition, original, learning(), seed=7)
    result.overlay.multipliers[0] = 0.25

    assert original.overlay.multipliers[0] == 1.0
    assert result.condition == condition
    assert result.rewired is (condition == "rewired_control")
    if condition == "rewired_control":
        assert result.post_ids != (20, 21, 22)


def test_controls_have_expected_effects() -> None:
    original = binding()
    no_plasticity = build_condition("no_plasticity", original, learning(), seed=7)
    kc_lesion = build_condition("kc_mbon_lesion", original, learning(), seed=7)
    dan_lesion = build_condition("dan_lesion", original, learning(), seed=7)
    assert no_plasticity.overlay.multipliers.tolist() == [1.0, 1.0, 1.0]
    assert kc_lesion.overlay.multipliers.tolist() == [0.0, 0.0, 0.0]
    assert dan_lesion.dan_enabled is False
