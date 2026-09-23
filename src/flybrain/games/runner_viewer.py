"""Local real-time viewer and human controls for the procedural runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import DQN

from flybrain.games.runner_env import RunnerConfig, RunnerEnv, RunnerObservation
from flybrain.games.runner_learning import load_runner_model


class ViewerConfig(BaseModel, frozen=True):
    """Runtime options for the local runner window."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: Path | None = None
    seed: int = Field(default=7, ge=0)
    width: int = Field(default=960, ge=320, le=3_840)
    height: int = Field(default=540, ge=180, le=2_160)
    fps: int = Field(default=60, ge=10, le=240)
    human_control: bool = False
    runner: RunnerConfig = Field(default_factory=RunnerConfig)


class ViewerSmokeResult(BaseModel, frozen=True):
    """Headless proof that the viewer pipeline renders valid frames."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = "runner-viewer-smoke-v1"
    frames: int = Field(ge=1)
    frame_shape: tuple[int, int, int]
    actions: int = Field(ge=1, le=2)
    finite_observations: bool
    final_distance: float = Field(ge=0.0)


def _agent_action(model: DQN | None, observation: RunnerObservation) -> int:
    if model is not None:
        prediction, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(prediction).item())
    distance = float(observation[4])
    grounded = bool(observation[2] > 0.5)
    return int(grounded and 0.02 < distance < 0.30)


def run_runner_viewer_smoke(
    *,
    steps: int,
    seed: int = 7,
    checkpoint: Path | None = None,
) -> ViewerSmokeResult:
    """Exercise policy, environment, and RGB generation without opening a window."""

    if steps < 1:
        raise ValueError("viewer smoke requires at least one step")
    env = RunnerEnv()
    model = load_runner_model(checkpoint, env) if checkpoint is not None else None
    observation, info = env.reset(seed=seed, options={"mode": "play"})
    actions: set[int] = set()
    finite = bool(np.all(np.isfinite(observation)))
    frame = env.render_rgb()
    for _ in range(steps):
        action = _agent_action(model, observation)
        actions.add(action)
        observation, _, terminated, truncated, info = env.step(action)
        finite = finite and bool(np.all(np.isfinite(observation)))
        frame = env.render_rgb()
        if terminated or truncated:
            observation, info = env.reset(seed=seed, options={"mode": "play"})
    return ViewerSmokeResult(
        frames=steps,
        frame_shape=frame.shape,
        actions=len(actions),
        finite_observations=finite,
        final_distance=float(info["distance"]),
    )


def run_runner_viewer(config: ViewerConfig) -> int:
    """Open a persistent Pygame window for agent playback or manual play."""

    import pygame

    pygame.init()
    try:
        screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("FlyBrain Game Learning — Runner")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 21)
        env = RunnerEnv(config.runner)
        model = load_runner_model(config.checkpoint, env) if config.checkpoint else None
        seed = config.seed
        observation, info = env.reset(seed=seed, options={"mode": "play"})
        human = config.human_control
        paused = False
        speed_values = (0.25, 0.5, 1.0, 2.0, 4.0)
        speed_index = 2
        accumulator = 0.0
        jump_requested = False
        episode_return = 0.0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        jump_requested = True
                    elif event.key == pygame.K_a:
                        human = not human
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        observation, info = env.reset(seed=seed, options={"mode": "play"})
                        episode_return = 0.0
                    elif event.key == pygame.K_n:
                        seed += 1
                        observation, info = env.reset(seed=seed, options={"mode": "play"})
                        episode_return = 0.0
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_index = max(0, speed_index - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_index = min(len(speed_values) - 1, speed_index + 1)

            if not paused:
                accumulator += speed_values[speed_index]
                while accumulator >= 1.0:
                    action = int(jump_requested) if human else _agent_action(model, observation)
                    jump_requested = False
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    accumulator -= 1.0
                    if terminated or truncated:
                        observation, info = env.reset(seed=seed, options={"mode": "play"})
                        episode_return = 0.0
                        break

            frame = env.render_rgb(width=config.width, height=config.height)
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            lines = (
                f"mode: {'human' if human else 'agent'}   seed: {seed}   "
                f"speed: {speed_values[speed_index]:g}x   {'PAUSED' if paused else ''}",
                f"distance: {float(info['distance']):.2f}   return: {episode_return:.2f}   "
                f"checkpoint: {config.checkpoint.name if config.checkpoint else 'heuristic demo'}",
            )
            screen.blit(font.render(lines[0], True, (245, 247, 255)), (18, 16))
            screen.blit(small_font.render(lines[1], True, (207, 216, 238)), (18, 45))
            controls = (
                "SPACE jump | A human/agent | P pause | R reset | "
                "N next | [ ] speed | Q quit"
            )
            screen.blit(
                small_font.render(controls, True, (207, 216, 238)),
                (18, config.height - 30),
            )
            pygame.display.flip()
            clock.tick(config.fps)
        return 0
    finally:
        pygame.quit()
