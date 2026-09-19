import pytest

from flybrain.behavioral_perturbations import BodyPerturbation, apply_perturbation
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
