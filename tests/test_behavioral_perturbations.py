import pytest

from flybrain.behavioral_perturbations import (
    BehaviorVariant,
    BodyPerturbation,
    apply_perturbation,
    scale_behavior_variant,
    world_digest,
)
from flybrain.hexapod_body import HexapodParameters


def test_perturbation_scales_physics_without_changing_dt() -> None:
    parameters = HexapodParameters(dt_s=0.01)
    perturbed = apply_perturbation(
        parameters,
        BodyPerturbation(friction_scale=2.0, mass_scale=3.0, delay_steps=2),
    )

    assert perturbed.dt_s == parameters.dt_s
    assert perturbed.friction_coefficient == 0.6
    assert perturbed.body_mass_kg == 0.003


@pytest.mark.parametrize(
    "kwargs",
    [
        {"friction_scale": 0.0},
        {"mass_scale": -1.0},
        {"delay_steps": -1},
        {"damaged_legs": (6,)},
        {"damaged_legs": (1, 1)},
    ],
)
def test_perturbation_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BodyPerturbation(**kwargs)


def test_world_digest_tracks_environment_not_label_or_scoring_target() -> None:
    first = BehaviorVariant(
        name="food-a",
        evaluation_target="food",
        food_position_m=(1.0, 2.0),
        threat_position_m=(3.0, 4.0),
        initial_position_m=(0.0, 0.0),
    )
    renamed = first.model_copy(
        update={"name": "threat-b", "evaluation_target": "threat"}
    )
    shifted = first.model_copy(update={"initial_position_m": (0.1, 0.0)})

    assert world_digest(first) == world_digest(renamed)
    assert world_digest(first) != world_digest(shifted)


def test_scale_behavior_variant_changes_world_not_identity_or_physics() -> None:
    original = BehaviorVariant(
        name="unseen-food",
        evaluation_target="food",
        food_position_m=(0.25, 0.15),
        threat_position_m=(9.5, 9.5),
        initial_position_m=(0.02, -0.01),
        perturbation=BodyPerturbation(friction_scale=0.9, delay_steps=1),
    )

    scaled = scale_behavior_variant(original, 0.04)

    assert scaled.food_position_m == pytest.approx((0.01, 0.006))
    assert scaled.threat_position_m == pytest.approx((0.38, 0.38))
    assert scaled.initial_position_m == pytest.approx((0.0008, -0.0004))
    assert scaled.name == original.name
    assert scaled.evaluation_target == original.evaluation_target
    assert scaled.perturbation == original.perturbation
    assert original.food_position_m == (0.25, 0.15)
    assert world_digest(scaled) != world_digest(original)


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan"), float("inf"), 1.01])
def test_scale_behavior_variant_rejects_invalid_scale(scale: float) -> None:
    variant = BehaviorVariant(
        name="unseen-food", food_position_m=(0.25, 0.15),
        threat_position_m=(9.5, 9.5),
    )
    with pytest.raises(ValueError, match="arena scale"):
        scale_behavior_variant(variant, scale)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_world_geometry_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BehaviorVariant(
            name="invalid",
            food_position_m=(value, 0.0),
            threat_position_m=(1.0, 1.0),
        )
