"""Official FlyGym locomotion bridge for the practical autonomy controller."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.embodied_world import MotorCommand
from flybrain.flygym_backend import flygym_availability


class PracticalFlyGymResult(BaseModel, frozen=True):
    """Measured MuJoCo execution of high-level autonomous motor commands."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["flygym-mujoco-2.1.0"] = "flygym-mujoco-2.1.0"
    controller: Literal["official-hybrid-turning-controller"] = (
        "official-hybrid-turning-controller"
    )
    controlled_joint_dofs: int = Field(ge=1)
    adhesion_channels: int = Field(ge=0)
    commands_consumed: int = Field(ge=1)
    physics_steps: int = Field(ge=1)
    duration_s: float = Field(gt=0.0)
    initial_thorax_position_mm: tuple[float, float, float]
    final_thorax_position_mm: tuple[float, float, float]
    horizontal_displacement_mm: float = Field(ge=0.0)
    final_thorax_height_mm: float
    minimum_thorax_height_mm: float
    all_actions_finite: bool
    stable: bool
    video_path: str | None
    rendered_frames: int = Field(ge=0)


def motor_command_to_descending_signal(
    command: MotorCommand,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Translate planar thrust/turn into official left/right descending drive."""

    turn_gain = 0.6
    left = float(np.clip(command.forward - turn_gain * command.turn, -1.2, 1.2))
    right = float(np.clip(command.forward + turn_gain * command.turn, -1.2, 1.2))
    return np.asarray((left, right), dtype=np.float64)


def _vec3(values: Any) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def run_practical_flygym(
    commands: tuple[MotorCommand, ...],
    *,
    duration_per_command_s: float = 0.02,
    video_path: Path | None = None,
) -> PracticalFlyGymResult:
    """Drive the official 42-DOF FlyGym body with autonomous commands."""

    if not commands:
        raise ValueError("at least one practical motor command is required")
    if not math.isfinite(duration_per_command_s) or duration_per_command_s <= 0.0:
        raise ValueError("duration per command must be finite and positive")
    availability = flygym_availability()
    if not availability.available:
        raise RuntimeError(availability.reason)

    flygym = importlib.import_module("flygym")
    mujoco = importlib.import_module("mujoco")
    anatomy = importlib.import_module("flygym.anatomy")
    compose = importlib.import_module("flygym.compose")
    math_utils = importlib.import_module("flygym.utils.math")
    locomotion = importlib.import_module("flygym_demo.complex_terrain")

    fly = locomotion.make_locomotion_fly(
        name="flybrain_practical",
        add_adhesion=True,
    )
    camera = None
    if video_path is not None:
        video_path = video_path.resolve()
        if video_path.exists():
            raise ValueError(f"video output already exists: {video_path}")
        camera = fly.add_tracking_camera(
            name="practical_camera",
            pos_offset=(-0.5, -7.5, 0.0),
            rotation=math_utils.Rotation3D("euler", (1.57, 0.0, 0.0)),
            fovy=35.0,
        )
    world = compose.FlatGroundWorld()
    world.add_fly(
        fly,
        (0.0, 0.0, 0.8),
        math_utils.Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        bodysegs_with_ground_contact=anatomy.ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    simulation = flygym.Simulation(world)
    simulation.reset()
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    renderer = (
        simulation.set_renderer(
            [camera],
            camera_res=(240, 320),
            playback_speed=0.1,
            output_fps=25,
        )
        if camera is not None
        else None
    )
    dof_order = fly.get_actuated_jointdofs_order("position")
    controller = locomotion.HybridTurningController(
        timestep=simulation.timestep,
        preprogrammed_steps=locomotion.PreprogrammedSteps(),
        output_dof_order=dof_order,
    )
    thorax_index = fly.get_bodysegs_order().index(anatomy.BodySegment("c_thorax"))
    initial = np.asarray(
        simulation.get_body_positions(fly.name)[thorax_index],
        dtype=np.float64,
    ).copy()
    steps_per_command = max(1, round(duration_per_command_s / simulation.timestep))
    action_finite = True
    adhesion_channels = 0
    minimum_height = float(initial[2])
    physics_steps = 0
    for command in commands:
        descending = motor_command_to_descending_signal(command)
        for _ in range(steps_per_command):
            observation = locomotion.HybridControllerObservation.from_sim(
                simulation,
                fly.name,
            )
            action = controller.step(descending, observation)
            joint_angles = np.asarray(action.joint_angles, dtype=np.float64)
            adhesion = np.asarray(action.adhesion_onoff, dtype=np.bool_)
            action_finite = action_finite and bool(np.isfinite(joint_angles).all())
            adhesion_channels = int(adhesion.size)
            locomotion.apply_locomotion_action(simulation, fly.name, action)
            simulation.step()
            if renderer is not None:
                simulation.render_as_needed()
            physics_steps += 1
            position = simulation.get_body_positions(fly.name)[thorax_index]
            minimum_height = min(minimum_height, float(position[2]))
    final = np.asarray(
        simulation.get_body_positions(fly.name)[thorax_index],
        dtype=np.float64,
    ).copy()
    displacement = float(np.linalg.norm(final[:2] - initial[:2]))
    finite_state = bool(np.isfinite(initial).all() and np.isfinite(final).all())
    stable = finite_state and action_finite and float(final[2]) > 0.3
    rendered_frames = 0
    if renderer is not None and video_path is not None:
        assert camera is not None
        rendered_frames = len(renderer.frames[camera.name])
        renderer.save_video(video_path)
        renderer.close()
    return PracticalFlyGymResult(
        controlled_joint_dofs=len(dof_order),
        adhesion_channels=adhesion_channels,
        commands_consumed=len(commands),
        physics_steps=physics_steps,
        duration_s=physics_steps * simulation.timestep,
        initial_thorax_position_mm=_vec3(initial),
        final_thorax_position_mm=_vec3(final),
        horizontal_displacement_mm=displacement,
        final_thorax_height_mm=float(final[2]),
        minimum_thorax_height_mm=minimum_height,
        all_actions_finite=action_finite,
        stable=stable,
        video_path=str(video_path) if video_path is not None else None,
        rendered_frames=rendered_frames,
    )
