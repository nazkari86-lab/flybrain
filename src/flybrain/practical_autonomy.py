"""Fast hybrid autonomy for a practical, non-biological engineering demo."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody, MotorCommand, WorldStep
from flybrain.practical_flygym import PracticalFlyGymResult


class PracticalAutonomyConfig(BaseModel, frozen=True):
    """Small bounded train/evaluate configuration for the fast hybrid path."""

    model_config = ConfigDict(extra="forbid")

    training_episodes: int = Field(default=24, ge=1, le=10_000)
    evaluation_episodes: int = Field(default=12, ge=1, le=1_000)
    max_steps: int = Field(default=400, ge=10, le=100_000)
    seed: int = Field(default=7, ge=0)
    arena_size: float = Field(default=10.0, gt=4.0, le=1_000.0)
    learning_rate: float = Field(default=0.25, gt=0.0, le=1.0)
    discount: float = Field(default=0.92, ge=0.0, lt=1.0)
    exploration: float = Field(default=0.12, ge=0.0, le=1.0)


class PracticalTraceStep(BaseModel, frozen=True):
    """One compact body/action record from the representative held-out run."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    x: float
    y: float
    heading_rad: float
    food_distance: float = Field(ge=0.0)
    threat_distance: float = Field(ge=0.0)
    action: str
    command: tuple[float, float]
    leg_phases: tuple[float, float, float, float, float, float]
    reward: float
    food_contact: bool
    threat_contact: bool
    wall_contact: bool


class PracticalAutonomyResult(BaseModel, frozen=True):
    """Operational metrics without a biological-intelligence claim."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["practical-autonomy-v1"] = "practical-autonomy-v1"
    evidence_kind: Literal["engineering_demo"] = "engineering_demo"
    controller: Literal["hybrid-reflex-potential-q-v1"] = (
        "hybrid-reflex-potential-q-v1"
    )
    engineering_demo_only: Literal[True] = True
    full_biological_intelligence_claim_allowed: Literal[False] = False
    training_episodes: int = Field(ge=1)
    evaluation_episodes: int = Field(ge=1)
    training_updates: int = Field(ge=1)
    learned_states: int = Field(ge=1)
    food_successes: int = Field(ge=0)
    threat_contacts: int = Field(ge=0)
    wall_contacts: int = Field(ge=0)
    mean_steps_to_food: float | None
    six_leg_gait_active: bool
    representative_trace: tuple[PracticalTraceStep, ...]
    physics: PracticalFlyGymResult | None = None
    q_table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_Action = tuple[str, MotorCommand]
_ACTIONS: tuple[_Action, ...] = (
    ("hard_left", MotorCommand(forward=0.55, turn=1.0)),
    ("soft_left", MotorCommand(forward=0.9, turn=0.35)),
    ("straight", MotorCommand(forward=1.0, turn=0.0)),
    ("soft_right", MotorCommand(forward=0.9, turn=-0.35)),
    ("hard_right", MotorCommand(forward=0.55, turn=-1.0)),
    ("reverse", MotorCommand(forward=-1.0, turn=0.0)),
)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _polar(body: FlyBody, point: tuple[float, float]) -> tuple[float, float]:
    dx = point[0] - body.x
    dy = point[1] - body.y
    return math.hypot(dx, dy), _wrap(math.atan2(dy, dx) - body.heading_rad)


def _bin_bearing(value: float) -> int:
    return int(((value + math.pi) % (2.0 * math.pi)) / (math.pi / 4.0))


def _state(observation: WorldStep) -> tuple[int, ...]:
    food_distance, food_bearing = _polar(observation.body, observation.food_position)
    threat_distance, threat_bearing = _polar(observation.body, observation.threat_position)
    return (
        _bin_bearing(food_bearing),
        0 if food_distance < 1.0 else 1 if food_distance < 3.0 else 2,
        _bin_bearing(threat_bearing),
        0 if threat_distance < 1.0 else 1 if threat_distance < 2.5 else 2,
        int(observation.wall_contact),
        -1 if observation.body.forward_speed < -0.1 else int(observation.body.forward_speed > 0.1),
    )


def _desired_turn(observation: WorldStep, arena_size: float) -> tuple[float, bool]:
    body = observation.body
    food_distance, food_bearing = _polar(body, observation.food_position)
    threat_distance, threat_bearing = _polar(body, observation.threat_position)
    danger = threat_distance < 2.25
    desired_bearing = _wrap(threat_bearing + math.pi) if danger else food_bearing

    margin = 0.7
    near_edge = (
        body.x < margin
        or body.y < margin
        or body.x > arena_size - margin
        or body.y > arena_size - margin
    )
    if observation.wall_contact or near_edge:
        desired_bearing = _wrap(
            math.atan2(arena_size / 2.0 - body.y, arena_size / 2.0 - body.x)
            - body.heading_rad
        )
    target_rate = max(-4.0, min(4.0, 3.0 * desired_bearing))
    turn_acceleration = 0.42 * (target_rate - body.angular_speed)
    if food_distance < 0.5 and not danger:
        turn_acceleration *= 0.5
    return max(-1.0, min(1.0, turn_acceleration)), danger or near_edge


def _heuristic_scores(observation: WorldStep, arena_size: float) -> list[float]:
    desired_turn, safety_active = _desired_turn(observation, arena_size)
    scores = []
    for _, command in _ACTIONS:
        score = -2.5 * abs(command.turn - desired_turn)
        score += 0.7 * command.forward
        if safety_active and command.forward < 0.0:
            score += 0.25
        scores.append(score)
    return scores


def _leg_phases(step: int, dt_s: float) -> tuple[float, float, float, float, float, float]:
    phase = (step * dt_s * 5.0) % 1.0
    opposite = (phase + 0.5) % 1.0
    return phase, opposite, opposite, phase, phase, opposite


def _sample_point(rng: random.Random, size: float) -> tuple[float, float]:
    return rng.uniform(1.0, size - 1.0), rng.uniform(1.0, size - 1.0)


def _make_world(rng: random.Random, size: float) -> ArenaWorld:
    while True:
        body_point = _sample_point(rng, size)
        food = _sample_point(rng, size)
        threat = _sample_point(rng, size)
        if math.dist(body_point, food) < 2.0:
            continue
        if math.dist(body_point, threat) < 1.5:
            continue
        if math.dist(food, threat) < 1.25:
            continue
        return ArenaWorld(
            ArenaConfig(size, size, 0.1, 0.15),
            FlyBody(
                x=body_point[0],
                y=body_point[1],
                heading_rad=rng.uniform(-math.pi, math.pi),
                forward_speed=0.0,
                angular_speed=0.0,
                energy=1.0,
                leg_contacts=(False,) * 6,
            ),
            food=food,
            threat=threat,
        )


def _select_action(
    observation: WorldStep,
    values: Sequence[float],
    *,
    arena_size: float,
    rng: random.Random,
    exploration: float,
) -> int:
    if exploration > 0.0 and rng.random() < exploration:
        return rng.randrange(len(_ACTIONS))
    heuristic = _heuristic_scores(observation, arena_size)
    combined = [
        base + 0.08 * max(-8.0, min(8.0, learned))
        for base, learned in zip(heuristic, values, strict=True)
    ]
    return max(range(len(combined)), key=combined.__getitem__)


def _reward(before: WorldStep, after: WorldStep) -> float:
    old_food, _ = _polar(before.body, before.food_position)
    new_food, _ = _polar(after.body, after.food_position)
    old_threat, _ = _polar(before.body, before.threat_position)
    new_threat, _ = _polar(after.body, after.threat_position)
    value = 1.8 * (old_food - new_food) - 0.01
    if min(old_threat, new_threat) < 2.5:
        value += 1.2 * (new_threat - old_threat)
    if after.food_contact:
        value += 25.0
    if after.threat_contact:
        value -= 35.0
    if after.wall_contact:
        value -= 1.0
    return value


def _run_episode(
    world: ArenaWorld,
    q_table: dict[tuple[int, ...], list[float]],
    config: PracticalAutonomyConfig,
    rng: random.Random,
    *,
    learn: bool,
    capture_trace: bool,
) -> tuple[bool, int, int, int, int, tuple[PracticalTraceStep, ...]]:
    trace: list[PracticalTraceStep] = []
    threat_contacts = 0
    wall_contacts = 0
    updates = 0
    for step in range(config.max_steps):
        before = world.observe()
        state = _state(before)
        values = q_table.setdefault(state, [0.0] * len(_ACTIONS))
        action_index = _select_action(
            before,
            values,
            arena_size=config.arena_size,
            rng=rng,
            exploration=config.exploration if learn else 0.0,
        )
        action_name, command = _ACTIONS[action_index]
        after = world.step(command)
        reward = _reward(before, after)
        if learn:
            next_values = q_table.setdefault(_state(after), [0.0] * len(_ACTIONS))
            target = reward + config.discount * max(next_values)
            values[action_index] += config.learning_rate * (target - values[action_index])
            updates += 1
        threat_contacts += int(after.threat_contact)
        wall_contacts += int(after.wall_contact)
        if capture_trace:
            food_distance, _ = _polar(after.body, after.food_position)
            threat_distance, _ = _polar(after.body, after.threat_position)
            trace.append(
                PracticalTraceStep(
                    step=step,
                    x=after.body.x,
                    y=after.body.y,
                    heading_rad=after.body.heading_rad,
                    food_distance=food_distance,
                    threat_distance=threat_distance,
                    action=action_name,
                    command=(command.forward, command.turn),
                    leg_phases=_leg_phases(step, world.config.dt_s),
                    reward=reward,
                    food_contact=after.food_contact,
                    threat_contact=after.threat_contact,
                    wall_contact=after.wall_contact,
                )
            )
        if after.food_contact:
            return True, step + 1, threat_contacts, wall_contacts, updates, tuple(trace)
        if after.threat_contact:
            return False, step + 1, threat_contacts, wall_contacts, updates, tuple(trace)
    return False, config.max_steps, threat_contacts, wall_contacts, updates, tuple(trace)


def _q_digest(q_table: dict[tuple[int, ...], list[float]]) -> str:
    payload = [
        {"state": list(state), "values": values}
        for state, values in sorted(q_table.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_practical_autonomy(config: PracticalAutonomyConfig) -> PracticalAutonomyResult:
    """Train the residual policy, then evaluate it on unseen randomized arenas."""

    training_rng = random.Random(config.seed)
    q_table: dict[tuple[int, ...], list[float]] = {}
    updates = 0
    for _ in range(config.training_episodes):
        _, _, _, _, episode_updates, _ = _run_episode(
            _make_world(training_rng, config.arena_size),
            q_table,
            config,
            training_rng,
            learn=True,
            capture_trace=False,
        )
        updates += episode_updates

    evaluation_rng = random.Random(config.seed ^ 0x5F3759DF)
    successes = 0
    threat_contacts = 0
    wall_contacts = 0
    success_steps: list[int] = []
    representative: tuple[PracticalTraceStep, ...] = ()
    for episode in range(config.evaluation_episodes):
        success, steps, threats, walls, _, trace = _run_episode(
            _make_world(evaluation_rng, config.arena_size),
            q_table,
            config,
            evaluation_rng,
            learn=False,
            capture_trace=episode == 0,
        )
        successes += int(success)
        threat_contacts += threats
        wall_contacts += walls
        if success:
            success_steps.append(steps)
        if episode == 0:
            representative = trace
    return PracticalAutonomyResult(
        training_episodes=config.training_episodes,
        evaluation_episodes=config.evaluation_episodes,
        training_updates=updates,
        learned_states=len(q_table),
        food_successes=successes,
        threat_contacts=threat_contacts,
        wall_contacts=wall_contacts,
        mean_steps_to_food=(
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        six_leg_gait_active=bool(representative)
        and all(len(item.leg_phases) == 6 for item in representative),
        representative_trace=representative,
        q_table_sha256=_q_digest(q_table),
    )
