from flybrain.games.runner_env import RunnerConfig
from flybrain.games.runner_learning import (
    RunnerTrainingConfig,
    evaluate_random_runner,
    evaluate_runner,
    train_runner,
)


def test_trained_runner_beats_random_on_unseen_seeds(tmp_path) -> None:
    runner = RunnerConfig(level_length=8, max_steps=900)
    run = train_runner(
        RunnerTrainingConfig(
            total_steps=16_384,
            checkpoint_every=4_096,
            seed=7,
            runner=runner,
        ),
        tmp_path / "run",
    )
    seeds = (7001, 7002, 7003, 7004)

    learned = evaluate_runner(run.best, seeds=seeds)
    random = evaluate_random_runner(runner, seeds=seeds, policy_seed=7)

    assert learned.mean_normalized_distance > random.mean_normalized_distance
    assert learned.training_mutated is False
