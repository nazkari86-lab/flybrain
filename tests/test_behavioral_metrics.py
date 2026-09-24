import pytest

from flybrain.behavioral_metrics import (
    BootstrapInterval,
    EpisodeObservation,
    FoodMetrics,
    PairedComparison,
    bootstrap_interval,
    claim_gate,
    compare_conditions,
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
            initial_threat_distance=1.0,
            final_threat_distance=2.0,
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
            initial_threat_distance=1.0,
            final_threat_distance=3.0,
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
    assert threat.mean_distance_improvement == pytest.approx(1.5)


def test_condition_comparison_uses_threat_clearance_not_only_contact() -> None:
    normal = observations()
    control = tuple(
        item.model_copy(
            update={
                "initial_threat_distance": 1.0,
                "final_threat_distance": 1.0,
            }
        )
        for item in normal
    )

    comparison = compare_conditions(
        normal,
        control,
        control="no_plasticity",
        seed=7,
        samples=100,
    )

    assert comparison.threat_delta.mean == pytest.approx(1.5)
    assert comparison.paired_observations == 2


def test_condition_comparison_scores_each_holdout_only_on_its_declared_task() -> None:
    normal = (
        EpisodeObservation(
            evaluation_target="food",
            food_contact=False,
            threat_contact=False,
            initial_food_distance=2.0,
            final_food_distance=1.0,
            initial_threat_distance=1.0,
            final_threat_distance=0.0,
            horizon_steps=10,
        ),
        EpisodeObservation(
            evaluation_target="threat",
            food_contact=False,
            threat_contact=False,
            initial_food_distance=2.0,
            final_food_distance=3.0,
            initial_threat_distance=1.0,
            final_threat_distance=2.0,
            horizon_steps=10,
        ),
    )
    control = (
        normal[0].model_copy(update={"final_food_distance": 2.0}),
        normal[1].model_copy(update={"final_threat_distance": 1.0}),
    )

    comparison = compare_conditions(
        normal,
        control,
        control="no_plasticity",
        seed=7,
        samples=100,
    )

    assert comparison.food_delta.mean == pytest.approx(1.0)
    assert comparison.threat_delta.mean == pytest.approx(1.0)


def test_condition_comparison_bootstraps_independent_replicates() -> None:
    normal = observations() * 2
    control = tuple(
        item.model_copy(
            update={
                "initial_threat_distance": 1.0,
                "final_threat_distance": 1.0,
            }
        )
        for item in normal
    )

    comparison = compare_conditions(
        normal,
        control,
        control="no_plasticity",
        seed=7,
        samples=100,
        replicate_size=2,
    )

    assert comparison.paired_observations == 2


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
        paired_observations=3,
    )
    comparisons = {
        "no_plasticity": good,
        "dan_lesion": good.model_copy(update={"control": "dan_lesion"}),
        "kc_mbon_lesion": good.model_copy(update={"control": "kc_mbon_lesion"}),
        "rewired_control": good.model_copy(update={"control": "rewired_control"}),
    }
    assert claim_gate(comparisons, generalization_verified=True) is True
    assert claim_gate(comparisons, generalization_verified=False) is False
    assert claim_gate({"no_plasticity": good}, generalization_verified=True) is False

    under_replicated = good.model_copy(update={"paired_observations": 2})
    assert claim_gate({
        "no_plasticity": under_replicated,
        "dan_lesion": under_replicated.model_copy(update={"control": "dan_lesion"}),
        "kc_mbon_lesion": under_replicated.model_copy(update={"control": "kc_mbon_lesion"}),
        "rewired_control": under_replicated.model_copy(update={"control": "rewired_control"}),
    }, generalization_verified=True) is False

    uncertain = good.model_copy(
        update={
            "food_delta": BootstrapInterval(mean=0.1, low=-0.1, high=0.3, samples=100),
            "threat_delta": BootstrapInterval(mean=0.1, low=-0.1, high=0.3, samples=100),
        }
    )
    assert claim_gate({
        "no_plasticity": uncertain,
        "dan_lesion": uncertain.model_copy(update={"control": "dan_lesion"}),
        "kc_mbon_lesion": uncertain.model_copy(update={"control": "kc_mbon_lesion"}),
        "rewired_control": uncertain.model_copy(update={"control": "rewired_control"}),
    }, generalization_verified=True) is False


def test_claim_gate_requires_replay_and_graph_integrity() -> None:
    good = PairedComparison(
        control="no_plasticity",
        food_delta=BootstrapInterval(mean=1.0, low=0.5, high=1.5, samples=100),
        threat_delta=BootstrapInterval(mean=1.0, low=0.5, high=1.5, samples=100),
        paired_observations=3,
    )
    comparisons = {
        "no_plasticity": good,
        "dan_lesion": good.model_copy(update={"control": "dan_lesion"}),
        "kc_mbon_lesion": good.model_copy(update={"control": "kc_mbon_lesion"}),
        "rewired_control": good.model_copy(update={"control": "rewired_control"}),
    }

    assert claim_gate(
        comparisons, replay_exact=False, generalization_verified=True
    ) is False
    assert claim_gate(
        comparisons, graph_unchanged=False, generalization_verified=True
    ) is False
