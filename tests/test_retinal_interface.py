import dataclasses

import pytest

from flybrain.embodied_world import FlyBody
from flybrain.retinal_interface import (
    RetinalObservation,
    VisualDisc,
    VisualInterfaceEncoder,
    VisualInterfaceMap,
    observe_retina,
)

BODY = FlyBody(5.0, 5.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6)
OBSERVATION = RetinalObservation(0.8, 0.2, 0.4, 0.1, 0.7, 0.3, 0.9)
ENCODER = VisualInterfaceEncoder(
    VisualInterfaceMap((1, 2), (3, 4), (5,), (6,), (7, 8), (9,))
)


def mirror_y(scene: tuple[VisualDisc, ...], center_y: float) -> tuple[VisualDisc, ...]:
    return tuple(VisualDisc(disc.x, 2 * center_y - disc.y, disc.radius) for disc in scene)


def test_mirroring_scene_swaps_motion_banks() -> None:
    previous = (VisualDisc(8.0, 5.5, 0.4),)
    current = (VisualDisc(8.0, 6.0, 0.4),)
    left = observe_retina(BODY, previous, current)
    right = observe_retina(BODY, mirror_y(previous, 5.0), mirror_y(current, 5.0))
    assert left.left_motion == pytest.approx(right.right_motion)
    assert left.right_motion == pytest.approx(right.left_motion)


def test_looming_is_monotonic_with_angular_expansion() -> None:
    far = observe_retina(
        BODY, (VisualDisc(9.0, 5.0, 0.2),), (VisualDisc(8.0, 5.0, 0.2),)
    )
    near = observe_retina(
        BODY, (VisualDisc(8.0, 5.0, 0.2),), (VisualDisc(6.0, 5.0, 0.2),)
    )
    assert 0.0 < far.looming < near.looming <= 1.0


def test_photoreceptor_events_never_contain_motion_or_looming_channels() -> None:
    events = ENCODER.encode_photoreceptors(OBSERVATION, step=3)
    assert {event.channel for event in events} <= {
        "luminance_left", "luminance_right", "contrast_left", "contrast_right"
    }


def test_feature_calibration_is_separate_and_normalized_by_bank_size() -> None:
    events = ENCODER.encode_feature_calibration(OBSERVATION, step=2)
    assert {event.channel for event in events} == {
        "hs_optic_flow_left", "hs_optic_flow_right", "lc16_looming_left", "lc16_looming_right"
    }
    assert sum(events[0].voltages) == pytest.approx(68.75 * OBSERVATION.left_motion)


def test_observation_has_no_object_labels_or_coordinates() -> None:
    fields = {field.name for field in dataclasses.fields(RetinalObservation)}
    assert fields == {
        "left_luminance", "right_luminance", "left_contrast", "right_contrast",
        "left_motion", "right_motion", "looming",
    }


@pytest.mark.parametrize(
    "previous,current,error",
    [
        ((VisualDisc(8.0, 5.0, 0.2),), (), "equal length"),
        ((VisualDisc(5.1, 5.0, 0.2),), (VisualDisc(5.1, 5.0, 0.2),), "overlap"),
    ],
)
def test_invalid_scenes_fail_closed(previous, current, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        observe_retina(BODY, previous, current)


def test_mapping_rejects_overlap_and_bad_ids() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        VisualInterfaceEncoder(VisualInterfaceMap((1,), (1,), (2,), (3,), (4,), (5,)))
    with pytest.raises(ValueError, match="positive IDs"):
        VisualInterfaceEncoder(VisualInterfaceMap((0,), (1,), (2,), (3,), (4,), (5,)))
