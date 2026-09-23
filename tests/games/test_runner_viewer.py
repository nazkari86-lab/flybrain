from flybrain.games.runner_viewer import run_runner_viewer_smoke


def test_viewer_smoke_renders_and_steps() -> None:
    result = run_runner_viewer_smoke(steps=20, seed=7)

    assert result.frames == 20
    assert result.frame_shape == (540, 960, 3)
    assert result.actions in {1, 2}
    assert result.finite_observations is True
