import pytest

from flybrain.behavioral_metrics import (
    BootstrapInterval,
    EpisodeObservation,
    FoodMetrics,
    PairedComparison,
    bootstrap_interval,
    claim_gate,
    food_metrics,
    threat_metrics,
)


def observations() -> tuple[EpisodeObservation, ...]:
    return (
        EpisodeObservation(
            food_contact=True,
            threat_contact=False,
            initial_food_distance=2.0,
            final_food_distance=0.5,
            first_food_contact_step=3,
            first_threat_contact_step=None,
            time_to_clear_threat_steps=None,
            horizon_steps=10,
        ),
        EpisodeObservation(
            food_contact=False,
            threat_contact=False,
            initial_food_distance=2.0,
            final_food_distance=1.0,
            first_food_contact_step=None,
            first_threat_contact_step=None,
            time_to_clear_threat_steps=None,
            horizon_steps=10,
        ),
    )


def test_metrics_are_environment_observations() -> None:
    food = food_metrics(observations())
    threat = threat_metrics(observations())

    assert food.contact_rate == pytest.approx(0.5)
    assert food.mean_time_to_contact_steps == pytest.approx(7.0)
    assert food.mean_distance_improvement == pytest.approx(1.25)
    assert food == FoodMetrics(
        contact_rate=0.5,
        mean_time_to_contact_steps=7.0,
        mean_distance_improvement=1.25,
    )
    assert threat.contact_rate == 0.0
    assert threat.avoidance_rate == 1.0


def test_bootstrap_is_deterministic_and_validated() -> None:
    interval = bootstrap_interval((1.0, 2.0, 3.0), seed=7, samples=100)
    assert interval == bootstrap_interval((1.0, 2.0, 3.0), seed=7, samples=100)
    assert interval.mean == pytest.approx(2.0)
    assert interval.low <= interval.mean <= interval.high
    with pytest.raises(ValueError):
        bootstrap_interval((), seed=7, samples=100)


def test_claim_gate_requires_all_controls_and_both_tasks() -> None:
    good = PairedComparison(
        control="no_plasticity",
        food_delta=BootstrapInterval(mean=1.0, low=0.5, high=1.5, samples=100),
        threat_delta=BootstrapInterval(mean=1.0, low=0.5, high=1.5, samples=100),
    )
    assert claim_gate({
        "no_plasticity": good,
        "dan_lesion": good.model_copy(update={"control": "dan_lesion"}),
        "kc_mbon_lesion": good.model_copy(update={"control": "kc_mbon_lesion"}),
        "rewired_control": good.model_copy(update={"control": "rewired_control"}),
    }) is True
    assert claim_gate({"no_plasticity": good}) is False
