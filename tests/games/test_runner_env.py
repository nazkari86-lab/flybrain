import numpy as np

from flybrain.games.runner_env import RunnerConfig, RunnerEnv


def test_same_seed_produces_same_level_and_trace() -> None:
    first = RunnerEnv(RunnerConfig(level_length=12))
    second = RunnerEnv(RunnerConfig(level_length=12))
    first_obs, first_info = first.reset(seed=11, options={"mode": "training"})
    second_obs, second_info = second.reset(seed=11, options={"mode": "training"})

    assert first_info["level_signature"] == second_info["level_signature"]
    np.testing.assert_array_equal(first_obs, second_obs)
    for action in (0, 1, 0, 0, 0, 1):
        left = first.step(action)
        right = second.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_jump_is_grounded_and_observation_stays_bounded() -> None:
    env = RunnerEnv(RunnerConfig(level_length=4))
    env.reset(seed=3)

    observation, _, _, _, _ = env.step(1)
    assert observation[1] < 0.0
    second, _, _, _, _ = env.step(1)
    assert second[1] > observation[1]
    assert np.all(np.isfinite(second))
    assert env.observation_space.contains(second)


def test_rgb_render_matches_declared_shape() -> None:
    env = RunnerEnv(RunnerConfig(level_length=4))
    env.reset(seed=5, options={"mode": "holdout"})

    frame = env.render_rgb(width=320, height=180)

    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    assert int(frame.max()) > int(frame.min())
