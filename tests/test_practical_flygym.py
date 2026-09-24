from pathlib import Path

import pytest

from flybrain.embodied_world import MotorCommand
from flybrain.flygym_backend import flygym_availability
from flybrain.practical_flygym import run_practical_flygym


@pytest.mark.skipif(
    not flygym_availability().available,
    reason="FlyGym 2.1 physics extra is not installed",
)
def test_official_locomotion_controller_moves_all_leg_dofs_without_falling() -> None:
    result = run_practical_flygym(
        (MotorCommand(forward=1.0, turn=0.0),) * 5,
        duration_per_command_s=0.02,
    )

    assert result.backend == "flygym-mujoco-2.1.0"
    assert result.controller == "official-hybrid-turning-controller"
    assert result.controlled_joint_dofs == 42
    assert result.adhesion_channels == 6
    assert result.physics_steps == 1_000
    assert result.initial_thorax_position_mm[2] > 0.3
    assert result.minimum_thorax_height_mm > 0.3
    assert result.horizontal_displacement_mm > 0.1
    assert result.stable is True
    assert result.all_actions_finite is True


@pytest.mark.skipif(
    not flygym_availability().available,
    reason="FlyGym 2.1 physics extra is not installed",
)
def test_official_locomotion_controller_renders_follow_camera_video(
    tmp_path: Path,
) -> None:
    video = tmp_path / "fly.mp4"

    result = run_practical_flygym(
        (MotorCommand(forward=1.0, turn=0.0),) * 2,
        duration_per_command_s=0.02,
        video_path=video,
    )

    assert result.video_path == str(video.resolve())
    assert result.rendered_frames >= 2
    assert video.stat().st_size > 1_000
