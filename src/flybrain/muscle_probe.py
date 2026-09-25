"""Explicit, partial MaleCNS-to-FlyMimic left-front muscle assay mapping."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS

MUSCLE_TO_GROUP = {
    "LFC_sternal_anterior_rotator": "left_fore_thorax_coxa_anterior",
    "LFC_sternal_posterior_rotator": "left_fore_thorax_coxa_posterior",
    "LFTibia_flex_93434": "left_fore_tibia_flexor",
    "LFTibia_extensor_93932": "left_fore_tibia_extensor",
}
MIN_MUSCLE_CONTROL = 0.0001


def muscle_action(
    activations: Sequence[float], muscle_names: Sequence[str]
) -> NDArray[np.float64]:
    """Route four name-matched populations; leave all other LF muscles at minimum.

    This is a cross-sex, population-to-muscle model assumption, not a validated
    innervation or force law. The FlyMimic body is tethered and LF-only.
    """
    group_names = tuple(item[0] for item in CANONICAL_MOTOR_GROUPS)
    if len(activations) != len(group_names):
        raise ValueError(f"expected {len(group_names)} canonical motor activations")
    if any(not math.isfinite(float(value)) for value in activations):
        raise ValueError("motor activations must be finite")
    if any(not 0.0 <= float(value) <= 1.0 for value in activations):
        raise ValueError("motor activations must be in [0, 1]")
    if len(set(muscle_names)) != len(muscle_names):
        raise ValueError("duplicate muscle name")
    missing = sorted(set(MUSCLE_TO_GROUP) - set(muscle_names))
    if missing:
        raise ValueError(f"missing muscle target: {missing}")
    action = np.full(len(muscle_names), MIN_MUSCLE_CONTROL, dtype=np.float64)
    group_index = {name: index for index, name in enumerate(group_names)}
    muscle_index = {name: index for index, name in enumerate(muscle_names)}
    for muscle, group in MUSCLE_TO_GROUP.items():
        action[muscle_index[muscle]] = max(
            MIN_MUSCLE_CONTROL, float(activations[group_index[group]])
        )
    return action


def replay_muscle_trace(
    trace: Sequence[Sequence[float]], *, window_s: float = 0.001
) -> dict[str, object]:
    """Open-loop replay of one MaleCNS motor trace in the tethered LF muscle body.

    This deliberately does not feed muscle proprioception back into MaleCNS or
    decode food, threat, gait or reward. A result is a component-level assay.
    """
    if not trace:
        raise ValueError("muscle replay requires at least one activation window")
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("muscle replay window must be finite and positive")
    if importlib.metadata.version("flygym") != "2.1.0":
        raise RuntimeError("FlyGym 2.1.0 required for muscle replay")
    compose = importlib.import_module("flygym.compose")
    sim, fly = compose.build_musculoskeletal_simulation()
    muscle_names = tuple(fly.muscle_names)
    joint_names = tuple(fly.get_jointdofs_order())
    observed_joints = (
        "joint_LFCoxa_yaw",
        "joint_LFCoxa_pitch",
        "joint_LFTibia_pitch",
    )
    missing_joints = sorted(set(observed_joints) - set(joint_names))
    if missing_joints:
        raise ValueError(f"missing muscle-model joint: {missing_joints}")
    physics_dt_s = float(sim.mj_model.opt.timestep)
    substeps = round(window_s / physics_dt_s)
    if substeps < 1 or not math.isclose(substeps * physics_dt_s, window_s, abs_tol=1e-10):
        raise ValueError("motor window must be an integer number of muscle physics steps")
    initial = np.asarray(sim.get_joint_angles(fly.name), dtype=np.float64).copy()
    force_sum = np.zeros(len(muscle_names), dtype=np.float64)
    for activations in trace:
        action = muscle_action(activations, muscle_names)
        sim.set_actuator_inputs(fly.name, compose.ActuatorType.MUSCLE, action)
        for _ in range(substeps):
            sim.step()
        angles = np.asarray(sim.get_joint_angles(fly.name), dtype=np.float64)
        forces = np.asarray(
            sim.get_actuator_forces(fly.name, compose.ActuatorType.MUSCLE),
            dtype=np.float64,
        )
        if not np.isfinite(angles).all() or not np.isfinite(forces).all():
            raise ValueError("muscle-model state became nonfinite")
        force_sum += np.abs(forces)
    final = np.asarray(sim.get_joint_angles(fly.name), dtype=np.float64)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    muscle_index = {name: index for index, name in enumerate(muscle_names)}
    return {
        "protocol": "open-loop-malecns-to-flymimic-lf-v1",
        "closed_loop": False,
        "body_model": "FlyGym 2.1.0 experimental FlyMimic tethered left-front leg",
        "steps": len(trace),
        "window_s": window_s,
        "physics_timestep_s": physics_dt_s,
        "physics_substeps_per_window": substeps,
        "muscle_names": muscle_names,
        "mapped_muscles": MUSCLE_TO_GROUP,
        "unmapped_muscles_at_minimum": tuple(
            name for name in muscle_names if name not in MUSCLE_TO_GROUP
        ),
        "initial_joint_angles_rad": {
            name: float(initial[joint_index[name]]) for name in observed_joints
        },
        "final_joint_angles_rad": {
            name: float(final[joint_index[name]]) for name in observed_joints
        },
        "mean_abs_muscle_force": {
            name: float(force_sum[muscle_index[name]] / len(trace))
            for name in MUSCLE_TO_GROUP
        },
    }
