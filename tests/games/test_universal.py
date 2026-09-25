from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from flybrain.games.universal import (
    UniversalGameConfig,
    _prepare_env,
    evaluate_universal_game,
    train_universal_game,
)


def make_cartpole() -> gym.Env[Any, Any]:
    return gym.make("CartPole-v1")


def make_pendulum() -> gym.Env[Any, Any]:
    return gym.make("Pendulum-v1")


def small_config(total_steps: int) -> UniversalGameConfig:
    return UniversalGameConfig(
        total_steps=total_steps,
        checkpoint_every=32,
        seed=7,
        training_seed_count=2,
        validation_seed_count=2,
        holdout_seed_count=2,
        buffer_size=128,
        learning_starts=8,
        batch_size=8,
        evaluation_max_steps=50,
    )


def test_universal_rejects_continuous_action_space() -> None:
    with pytest.raises(ValueError, match="Discrete action space"):
        _prepare_env(make_pendulum)


def test_universal_checkpoint_evaluation_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "cartpole"
    first = train_universal_game(
        make_cartpole,
        env_name="CartPole-v1",
        config=small_config(64),
        output=output,
    )

    manifest = first.latest.joinpath("manifest.json").read_text(encoding="utf-8")
    assert '"protocol":"universal-gym-dqn-v2"' in manifest.replace(" ", "")
    assert first.latest.joinpath("model.zip").is_file()
    assert first.latest.joinpath("replay.pkl").is_file()
    assert first.best.joinpath("manifest.json").is_file()

    evaluation = evaluate_universal_game(first.best, make_cartpole)
    assert evaluation.episodes == 2
    assert evaluation.training_mutated is False
    assert evaluation.model_sha256

    resumed = train_universal_game(
        make_cartpole,
        env_name="CartPole-v1",
        config=small_config(96),
        output=output,
        resume=first.latest,
    )
    assert resumed.completed_steps == 96
    assert resumed.latest.joinpath("replay.pkl").is_file()
