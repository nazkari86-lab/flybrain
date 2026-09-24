import pytest

from flybrain.behavioral_perturbations import (
    BehaviorVariant,
    BodyPerturbation,
    apply_perturbation,
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_world_geometry_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BehaviorVariant(
            name="invalid",
            food_position_m=(value, 0.0),
            threat_position_m=(1.0, 1.0),
        )
