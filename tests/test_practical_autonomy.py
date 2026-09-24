from flybrain.practical_autonomy import PracticalAutonomyConfig, run_practical_autonomy


def test_hybrid_controller_learns_and_solves_held_out_arenas() -> None:
    result = run_practical_autonomy(
        PracticalAutonomyConfig(
            training_episodes=12,
            evaluation_episodes=6,
            max_steps=300,
            seed=7,
        )
    )

    assert result.controller == "hybrid-reflex-potential-q-v1"
    assert result.engineering_demo_only is True
    assert result.training_updates > 0
    assert result.learned_states > 0
    assert result.food_successes >= 5
    assert result.threat_contacts == 0
    assert result.six_leg_gait_active is True
    assert all(len(step.leg_phases) == 6 for step in result.representative_trace)


def test_practical_autonomy_is_reproducible_and_counts_every_update() -> None:
    config = PracticalAutonomyConfig(
        training_episodes=3,
        evaluation_episodes=2,
        max_steps=80,
        seed=19,
    )

    first = run_practical_autonomy(config)
    second = run_practical_autonomy(config)

    assert first == second
    assert first.training_updates >= config.training_episodes
    assert first.training_updates <= config.training_episodes * config.max_steps
