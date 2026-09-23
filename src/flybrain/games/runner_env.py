"""Deterministic procedural one-button runner for fast local RL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from flybrain.games.contracts import GameMode, StepResult

RunnerObservation = NDArray[np.float32]
RgbFrame = NDArray[np.uint8]
ObstacleKind = Literal["spike", "block", "gap"]


class RunnerConfig(BaseModel, frozen=True):
    """Bounded physics and generation parameters for one runner level."""

    model_config = ConfigDict(extra="forbid")

    level_length: int = Field(default=24, ge=1, le=1_000)
    max_steps: int = Field(default=3_600, ge=10, le=100_000)
    dt: float = Field(default=1.0 / 60.0, gt=0.0, le=0.1)
    speed: float = Field(default=6.0, gt=0.0, le=20.0)
    gravity: float = Field(default=32.0, gt=0.0, le=100.0)
    jump_velocity: float = Field(default=-11.0, ge=-30.0, lt=0.0)
    player_width: float = Field(default=0.55, gt=0.1, le=2.0)
    player_height: float = Field(default=0.75, gt=0.1, le=2.0)
    lookahead: float = Field(default=14.0, gt=1.0, le=100.0)
    default_seed: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class Obstacle:
    """One generated world hazard."""

    kind: ObstacleKind
    x: float
    width: float
    height: float


@dataclass(frozen=True)
class RunnerState:
    """Authoritative fixed-timestep state."""

    x: float = 0.0
    y: float = 0.0
    vertical_velocity: float = 0.0
    grounded: bool = True
    step: int = 0
    obstacles_passed: int = 0
    terminated: bool = False


_KIND_VALUE: dict[ObstacleKind, float] = {"spike": -1.0, "block": 0.0, "gap": 1.0}


class RunnerEnv(gym.Env[RunnerObservation, int]):
    """Seeded Gymnasium environment with binary wait/jump actions."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__()
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
        self.config = config or RunnerConfig()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )
        self._seed = self.config.default_seed
        self._mode: GameMode = "training"
        self._difficulty = 0.0
        self._obstacles: tuple[Obstacle, ...] = ()
        self._level_signature = ""
        self._level_end = 1.0
        self._state = RunnerState()

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def obstacles(self) -> tuple[Obstacle, ...]:
        return self._obstacles

    @property
    def level_signature(self) -> str:
        return self._level_signature

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[RunnerObservation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = self.config.default_seed if seed is None else seed
        supplied = options or {}
        mode = supplied.get("mode", "training")
        if mode not in {"training", "validation", "holdout", "play"}:
            raise ValueError(f"unsupported runner mode: {mode}")
        self._mode = mode
        self._difficulty = float(np.clip(supplied.get("difficulty", 0.0), 0.0, 1.0))
        self._obstacles = self._generate_level(self._seed, self._difficulty)
        self._level_end = self._obstacles[-1].x + self._obstacles[-1].width + 4.0
        self._level_signature = self._signature(self._obstacles)
        self._state = RunnerState()
        return self._observation(), self._info(collision=False, completed=False)

    def step(
        self,
        action: int,
    ) -> tuple[RunnerObservation, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid runner action: {action}")
        if self._state.terminated:
            raise RuntimeError("runner episode is terminated; call reset")

        previous = self._state
        vertical_velocity = previous.vertical_velocity
        grounded = previous.grounded
        if action == 1 and grounded:
            vertical_velocity = self.config.jump_velocity
            grounded = False
        if not grounded:
            vertical_velocity += self.config.gravity * self.config.dt
        y = previous.y + vertical_velocity * self.config.dt
        if y >= 0.0:
            y = 0.0
            vertical_velocity = 0.0
            grounded = True

        x = previous.x + self.config.speed * self.config.dt
        passed = sum(
            obstacle.x + obstacle.width < x - self.config.player_width / 2.0
            for obstacle in self._obstacles
        )
        collision = self._collides(x=x, y=y)
        completed = x >= self._level_end
        terminated = collision or completed
        step = previous.step + 1
        truncated = step >= self.config.max_steps and not terminated
        self._state = RunnerState(
            x=x,
            y=y,
            vertical_velocity=vertical_velocity,
            grounded=grounded,
            step=step,
            obstacles_passed=passed,
            terminated=terminated or truncated,
        )

        reward = x - previous.x
        reward += 5.0 * (passed - previous.obstacles_passed)
        reward -= 0.002 * int(action == 1)
        if collision:
            reward -= 25.0
        if completed:
            reward += 100.0
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(collision=collision, completed=completed),
        )

    def render(self) -> RgbFrame:
        return self.render_rgb()

    def render_rgb(self, *, width: int = 960, height: int = 540) -> RgbFrame:
        if width < 64 or height < 64:
            raise ValueError("runner frame dimensions must be at least 64 pixels")
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = np.asarray((18, 24, 45), dtype=np.uint8)
        ground_y = int(height * 0.78)
        frame[ground_y:, :] = np.asarray((42, 50, 70), dtype=np.uint8)
        scale = width / 16.0
        camera_x = max(0.0, self._state.x - 3.0)

        for obstacle in self._obstacles:
            left = int((obstacle.x - camera_x) * scale)
            right = int((obstacle.x + obstacle.width - camera_x) * scale)
            if right < 0 or left >= width:
                continue
            left = max(0, left)
            right = min(width, max(left + 1, right))
            obstacle_height = max(2, int(obstacle.height * scale))
            if obstacle.kind == "gap":
                frame[ground_y:, left:right] = np.asarray((3, 5, 12), dtype=np.uint8)
            elif obstacle.kind == "block":
                frame[max(0, ground_y - obstacle_height) : ground_y, left:right] = np.asarray(
                    (75, 184, 205), dtype=np.uint8
                )
            else:
                center = (left + right) // 2
                half_width = max(1, (right - left) // 2)
                for row in range(obstacle_height):
                    fraction = row / obstacle_height
                    span = max(1, int(half_width * fraction))
                    y_index = ground_y - obstacle_height + row
                    frame[y_index, max(0, center - span) : min(width, center + span + 1)] = (
                        247,
                        84,
                        112,
                    )

        player_x = int(3.0 * scale)
        player_bottom = int(ground_y + self._state.y * scale)
        player_width = max(2, int(self.config.player_width * scale))
        player_height = max(2, int(self.config.player_height * scale))
        frame[
            max(0, player_bottom - player_height) : min(height, player_bottom),
            max(0, player_x - player_width // 2) : min(width, player_x + player_width // 2),
        ] = np.asarray((250, 210, 82), dtype=np.uint8)
        return frame

    def _generate_level(self, seed: int, difficulty: float) -> tuple[Obstacle, ...]:
        generator = np.random.default_rng(seed)
        obstacles: list[Obstacle] = []
        x = 5.0
        for _ in range(self.config.level_length):
            gap = float(generator.uniform(3.2 - difficulty * 0.6, 5.6 - difficulty * 0.8))
            x += gap
            kind: ObstacleKind = ("spike", "block", "gap")[int(generator.integers(0, 3))]
            width = float(generator.uniform(0.45, 0.85 + difficulty * 0.35))
            height = float(generator.uniform(0.65, 1.1 + difficulty * 0.35))
            obstacles.append(Obstacle(kind=kind, x=x, width=width, height=height))
            x += width
        return tuple(obstacles)

    @staticmethod
    def _signature(obstacles: tuple[Obstacle, ...]) -> str:
        payload = [
            {"height": item.height, "kind": item.kind, "width": item.width, "x": item.x}
            for item in obstacles
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _collides(self, *, x: float, y: float) -> bool:
        player_left = x - self.config.player_width / 2.0
        player_right = x + self.config.player_width / 2.0
        for obstacle in self._obstacles:
            overlaps = player_right >= obstacle.x and player_left <= obstacle.x + obstacle.width
            if not overlaps:
                continue
            if obstacle.kind == "gap":
                return y > -0.35
            clearance = obstacle.height + self.config.player_height * 0.15
            if y > -clearance:
                return True
        return False

    def _observation(self) -> RunnerObservation:
        upcoming = [
            obstacle
            for obstacle in self._obstacles
            if obstacle.x + obstacle.width >= self._state.x - self.config.player_width / 2.0
        ]

        def features(index: int) -> tuple[float, float, float]:
            if index >= len(upcoming):
                return 1.0, 0.0, 0.0
            obstacle = upcoming[index]
            distance = (obstacle.x - self._state.x) / self.config.lookahead
            return (
                float(np.clip(distance, -1.0, 1.0)),
                _KIND_VALUE[obstacle.kind],
                float(np.clip(obstacle.width / 2.0, 0.0, 1.0)),
            )

        first_distance, first_kind, first_width = features(0)
        second_distance, second_kind, _ = features(1)
        return np.asarray(
            (
                np.clip(self._state.y / 4.0, -1.0, 1.0),
                np.clip(self._state.vertical_velocity / 12.0, -1.0, 1.0),
                float(self._state.grounded),
                np.clip(self.config.speed / 20.0, 0.0, 1.0),
                first_distance,
                first_kind,
                first_width,
                second_distance,
                second_kind,
            ),
            dtype=np.float32,
        )

    def _info(self, *, collision: bool, completed: bool) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "mode": self._mode,
            "level_signature": self._level_signature,
            "distance": self._state.x,
            "normalized_distance": min(1.0, self._state.x / self._level_end),
            "obstacles_passed": self._state.obstacles_passed,
            "collision": collision,
            "completed": completed,
        }


class RunnerGameAdapter:
    """Shared-contract wrapper around the Gymnasium environment."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.env = RunnerEnv(config)

    def reset(self, *, seed: int, mode: GameMode) -> RunnerObservation:
        observation, _ = self.env.reset(seed=seed, options={"mode": mode})
        return observation

    def legal_actions(self) -> tuple[int, ...]:
        return (0, 1)

    def step(self, action: int) -> StepResult[RunnerObservation]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        shared_info = {
            key: value
            for key, value in info.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=shared_info,
        )
