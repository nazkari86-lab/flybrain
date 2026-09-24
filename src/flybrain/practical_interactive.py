"""Keyboard-driven real-time MuJoCo application for the practical fly."""

from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.embodied_world import MotorCommand
from flybrain.flygym_backend import flygym_availability
from flybrain.practical_flygym import motor_command_to_descending_signal


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass
class InteractiveControls:
    """Persistent keyboard command state independent of the GUI backend."""

    forward: float = 0.0
    turn: float = 0.0
    autopilot: bool = False
    quit_requested: bool = False
    _reset_requested: bool = False

    def handle_key(self, keycode: int) -> None:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W":
            self.forward = _clamp(round(self.forward + 0.2, 10))
            self.autopilot = False
        elif key == "S":
            self.forward = _clamp(round(self.forward - 0.2, 10))
            self.autopilot = False
        elif key == "A":
            self.turn = _clamp(round(self.turn + 0.2, 10))
            self.autopilot = False
        elif key == "D":
            self.turn = _clamp(round(self.turn - 0.2, 10))
            self.autopilot = False
        elif key == "C":
            self.turn = 0.0
            self.autopilot = False
        elif keycode == ord(" "):
            self.forward = 0.0
            self.turn = 0.0
            self.autopilot = False
        elif key == "P":
            self.autopilot = not self.autopilot
        elif key == "R":
            self._reset_requested = True
        elif key == "Q" or keycode == 27:
            self.quit_requested = True

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def command(self, simulation_time_s: float) -> MotorCommand:
        if self.autopilot:
            return MotorCommand(
                forward=1.0,
                turn=0.65 * math.sin(simulation_time_s * 1.4),
            )
        return MotorCommand(forward=self.forward, turn=self.turn)

    def overlay(self, simulation_time_s: float) -> tuple[str, str]:
        command = self.command(simulation_time_s)
        mode = "AUTOPILOT" if self.autopilot else "MANUAL"
        left = (
            "FlyBrain interactive\n"
            "W/S  speed +/-\n"
            "A/D  turn left/right\n"
            "C    center steering\n"
            "SPACE stop\n"
            "R reset   P autopilot   Q quit"
        )
        right = (
            f"mode: {mode}\n"
            f"forward: {command.forward:+.2f}\n"
            f"turn: {command.turn:+.2f}\n"
            f"time: {simulation_time_s:.2f} s"
        )
        return left, right


@dataclass
class _InteractiveSession:
    fly: Any
    simulation: Any
    controller: Any
    locomotion: Any
    mujoco: Any
    thorax_index: int
    thorax_body_id: int
    controlled_joint_dofs: int


class InteractiveSmokeResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    controlled_joint_dofs: int = Field(ge=1)
    adhesion_channels: int = Field(ge=0)
    steps: int = Field(ge=1)
    all_actions_finite: bool
    horizontal_displacement_mm: float = Field(ge=0.0)


def _build_session() -> _InteractiveSession:
    availability = flygym_availability()
    if not availability.available:
        raise RuntimeError(availability.reason)
    flygym = importlib.import_module("flygym")
    anatomy = importlib.import_module("flygym.anatomy")
    compose = importlib.import_module("flygym.compose")
    math_utils = importlib.import_module("flygym.utils.math")
    locomotion = importlib.import_module("flygym_demo.complex_terrain")
    mujoco = importlib.import_module("mujoco")

    fly = locomotion.make_locomotion_fly(
        name="flybrain_interactive",
        add_adhesion=True,
        colorize=True,
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
    dof_order = fly.get_actuated_jointdofs_order("position")
    controller = locomotion.HybridTurningController(
        timestep=simulation.timestep,
        preprogrammed_steps=locomotion.PreprogrammedSteps(),
        output_dof_order=dof_order,
    )
    thorax_index = fly.get_bodysegs_order().index(anatomy.BodySegment("c_thorax"))
    thorax_body_id = simulation._internal_bodyids_by_fly[fly.name][thorax_index]
    return _InteractiveSession(
        fly=fly,
        simulation=simulation,
        controller=controller,
        locomotion=locomotion,
        mujoco=mujoco,
        thorax_index=thorax_index,
        thorax_body_id=thorax_body_id,
        controlled_joint_dofs=len(dof_order),
    )


def _step_session(session: _InteractiveSession, command: MotorCommand) -> tuple[bool, int]:
    observation = session.locomotion.HybridControllerObservation.from_sim(
        session.simulation,
        session.fly.name,
    )
    action = session.controller.step(
        motor_command_to_descending_signal(command),
        observation,
    )
    joint_angles = np.asarray(action.joint_angles, dtype=np.float64)
    adhesion = np.asarray(action.adhesion_onoff, dtype=np.bool_)
    session.locomotion.apply_locomotion_action(
        session.simulation,
        session.fly.name,
        action,
    )
    session.simulation.step()
    return bool(np.isfinite(joint_angles).all()), int(adhesion.size)


def _reset_session(session: _InteractiveSession) -> None:
    session.simulation.reset()
    session.mujoco.mj_forward(
        session.simulation.mj_model,
        session.simulation.mj_data,
    )
    session.controller.reset(seed=0)


def run_interactive_smoke(*, steps: int = 200) -> InteractiveSmokeResult:
    if type(steps) is not int or steps <= 0:
        raise ValueError("interactive smoke steps must be a positive integer")
    session = _build_session()
    initial = np.asarray(
        session.simulation.get_body_positions(session.fly.name)[session.thorax_index],
        dtype=np.float64,
    )
    finite = True
    adhesion_channels = 0
    command = MotorCommand(forward=1.0, turn=0.0)
    for _ in range(steps):
        action_finite, adhesion_channels = _step_session(session, command)
        finite = finite and action_finite
    final = np.asarray(
        session.simulation.get_body_positions(session.fly.name)[session.thorax_index],
        dtype=np.float64,
    )
    return InteractiveSmokeResult(
        controlled_joint_dofs=session.controlled_joint_dofs,
        adhesion_channels=adhesion_channels,
        steps=steps,
        all_actions_finite=finite,
        horizontal_displacement_mm=float(np.linalg.norm(final[:2] - initial[:2])),
    )


def run_interactive_viewer() -> None:
    """Open the live MuJoCo window and run until the user closes or quits."""

    session = _build_session()
    viewer_module = importlib.import_module("mujoco.viewer")
    controls = InteractiveControls()
    viewer = viewer_module.launch_passive(
        session.simulation.mj_model,
        session.simulation.mj_data,
        key_callback=controls.handle_key,
        show_left_ui=False,
        show_right_ui=True,
    )
    viewer.cam.type = session.mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = session.thorax_body_id
    viewer.cam.distance = 8.0
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -20.0
    sync_interval = max(1, round(0.005 / session.simulation.timestep))
    step_index = 0
    wall_origin = time.perf_counter()
    simulation_origin = float(session.simulation.mj_data.time)
    try:
        while viewer.is_running() and not controls.quit_requested:
            if controls.consume_reset_request():
                _reset_session(session)
                wall_origin = time.perf_counter()
                simulation_origin = float(session.simulation.mj_data.time)
            command = controls.command(float(session.simulation.mj_data.time))
            _step_session(session, command)
            step_index += 1
            if step_index % sync_interval == 0:
                left, right = controls.overlay(float(session.simulation.mj_data.time))
                viewer.set_texts((None, None, left, right))
                viewer.sync()
                target = wall_origin + (
                    float(session.simulation.mj_data.time) - simulation_origin
                )
                delay = target - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        viewer.close()


def launch_interactive_process() -> int:
    """Launch through mjpython on macOS, or directly on other platforms."""

    if sys.platform != "darwin":
        run_interactive_viewer()
        return 0
    candidate = Path(sys.executable).with_name("mjpython")
    executable = str(candidate) if candidate.is_file() else shutil.which("mjpython")
    if executable is None:
        raise RuntimeError("mjpython is required for the interactive viewer on macOS")
    environment = os.environ.copy()
    completed = subprocess.run(
        [executable, "-m", "flybrain.practical_interactive"],
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main() -> None:
    run_interactive_viewer()


if __name__ == "__main__":
    main()
