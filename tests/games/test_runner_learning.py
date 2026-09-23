from flybrain.games.runner_env import RunnerConfig
from flybrain.games.runner_learning import (
    RunnerTrainingConfig,
    evaluate_random_runner,
    evaluate_runner,
    evaluate_untrained_runner,
    train_runner,
)


def test_short_training_writes_resumable_checkpoint(tmp_path) -> None:
    run = train_runner(
        RunnerTrainingConfig(total_steps=256, checkpoint_every=128, seed=7),
        tmp_path / "run",
    )

    assert run.latest.joinpath("manifest.json").is_file()
    assert run.latest.joinpath("model.zip").is_file()
    result = evaluate_runner(run.latest, seeds=(101, 102))
    assert result.episodes == 2
    assert result.model_sha256 == run.model_sha256
    assert result.training_mutated is False


def test_resume_increases_steps_without_overwriting_prior_checkpoint(tmp_path) -> None:
    first = train_runner(
        RunnerTrainingConfig(total_steps=128, checkpoint_every=128, seed=13),
        tmp_path / "run",
    )
    old_checkpoint = first.latest.resolve()
    old_hash = first.model_sha256

    resumed = train_runner(
        RunnerTrainingConfig(total_steps=256, checkpoint_every=128, seed=13),
        tmp_path / "run",
        resume=first.latest,
    )

    assert resumed.completed_steps >= 256
    assert old_checkpoint.joinpath("model.zip").is_file()
    assert resumed.latest.resolve() != old_checkpoint
    assert resumed.model_sha256 != old_hash


def test_baselines_are_reproducible_and_have_no_training_mutation() -> None:
    seeds = (201, 202)
    config = RunnerConfig(level_length=4, max_steps=400)

    first = evaluate_random_runner(config, seeds=seeds, policy_seed=17)
    second = evaluate_random_runner(config, seeds=seeds, policy_seed=17)
    untrained = evaluate_untrained_runner(config, seeds=seeds, policy_seed=17)

    assert first == second
    assert first.training_mutated is False
    assert untrained.training_mutated is False
    assert first.episodes == untrained.episodes == 2
