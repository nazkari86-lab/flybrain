"""Universal Gymnasium game connection, training, and live playback.

The connector deliberately keeps the game boundary small: a caller supplies a
Gymnasium environment factory, and FlyBrain owns the seeded training loop,
checkpoint identity, and live observation of the same environment interface.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from flybrain.games.artifacts import publish_directory_atomic, write_json_atomic

EnvFactory = Callable[[], gym.Env[Any, Any]]
SnapshotCallback = Callable[[Path, "UniversalCheckpointManifest"], None]


class UniversalGameConfig(BaseModel, frozen=True):
    """Reproducible configuration for one generic discrete-action game."""

    model_config = ConfigDict(extra="forbid")

    total_steps: int = Field(default=100_000, ge=1, le=100_000_000)
    checkpoint_every: int = Field(default=10_000, ge=1, le=10_000_000)
    seed: int = Field(default=7, ge=0)
    buffer_size: int = Field(default=50_000, ge=128, le=5_000_000)
    learning_starts: int = Field(default=500, ge=0, le=5_000_000)
    batch_size: int = Field(default=64, ge=1, le=4096)


class UniversalCheckpointManifest(BaseModel, frozen=True):
    """Identity and measured progress for one generic-game checkpoint."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = "universal-gym-dqn-v1"
    env_name: str = Field(min_length=1)
    completed_steps: int = Field(ge=0)
    episodes: int = Field(ge=0)
    action_count: int = Field(ge=2)
    observation_size: int = Field(ge=1)
    recent_mean_reward: float
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: dict[str, str]


class UniversalGameRun(BaseModel, frozen=True):
    """Paths and progress returned by a completed connection run."""

    model_config = ConfigDict(extra="forbid")

    output: Path
    latest: Path
    best: Path
    completed_steps: int = Field(ge=1)
    checkpoints: int = Field(ge=1)


@dataclass(frozen=True)
class _FactorySpec:
    factory: EnvFactory
    name: str


def load_factory(spec: str) -> _FactorySpec:
    """Load a local ``module:callable`` environment factory."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use the module:callable form")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"factory is not callable: {spec}")
    return _FactorySpec(factory=factory, name=spec)


def env_id_factory(env_id: str) -> _FactorySpec:
    """Create a connector for any registered Gymnasium environment."""

    if not env_id.strip():
        raise ValueError("env_id must not be empty")

    def factory() -> gym.Env[Any, Any]:
        try:
            return gym.make(env_id, render_mode="rgb_array")
        except TypeError:
            return gym.make(env_id)

    return _FactorySpec(factory=factory, name=env_id)


def _prepare_env(factory: EnvFactory, *, seed: int | None = None) -> gym.Env[Any, Any]:
    env = factory()
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError(
            "universal DQN requires a Discrete action space; provide a custom learner "
            "adapter for continuous or structured actions"
        )
    if not isinstance(env.observation_space, gym.spaces.Space):
        env.close()
        raise ValueError("factory returned an invalid Gymnasium observation space")
    prepared: gym.Env[Any, Any] = env
    if (
        not isinstance(env.observation_space, gym.spaces.Box)
        or len(env.observation_space.shape) != 1
    ):
        prepared = FlattenObservation(prepared)
    prepared = RecordEpisodeStatistics(prepared)
    if seed is not None:
        prepared.reset(seed=seed)
    return prepared


def _dependencies() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("gymnasium", "numpy", "stable-baselines3", "torch", "pygame-ce")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_directory_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=link.parent) as temporary:
        staged = Path(temporary) / link.name
        staged.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        os.replace(staged, link)


class _CheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        output: Path,
        env_name: str,
        config: UniversalGameConfig,
        action_count: int,
        observation_size: int,
        on_snapshot: SnapshotCallback | None,
    ) -> None:
        super().__init__(verbose=0)
        self.output = output
        self.env_name = env_name
        self.config = config
        self.action_count = action_count
        self.observation_size = observation_size
        self.on_snapshot = on_snapshot
        self.checkpoints = 0
        self.episodes = 0
        self.rewards: deque[float] = deque(maxlen=100)
        self.best_score = float("-inf")
        self.latest: Path | None = None
        self.best: Path | None = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if isinstance(episode, dict) and "r" in episode:
                self.episodes += 1
                self.rewards.append(float(np.asarray(episode["r"]).reshape(-1)[0]))
        if self.num_timesteps > 0 and (
            self.num_timesteps % self.config.checkpoint_every == 0
            or self.num_timesteps >= self.config.total_steps
        ):
            self._publish()
        return True

    def _publish(self) -> None:
        step = int(self.num_timesteps)
        destination = self.output / "checkpoints" / f"step-{step:012d}"
        score = float(np.mean(self.rewards)) if self.rewards else 0.0

        def writer(stage: Path) -> None:
            self.model.save(stage / "model")
            model_path = stage / "model.zip"
            manifest = UniversalCheckpointManifest(
                env_name=self.env_name,
                completed_steps=step,
                episodes=self.episodes,
                action_count=self.action_count,
                observation_size=self.observation_size,
                recent_mean_reward=score,
                model_sha256=_sha256(model_path),
                dependencies=_dependencies(),
            )
            stage.joinpath("manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

        checkpoint = publish_directory_atomic(destination, writer)
        self.latest = checkpoint
        if self.best is None or score >= self.best_score:
            self.best_score = score
            self.best = checkpoint
        _replace_directory_link(self.output / "latest", self.latest)
        _replace_directory_link(self.output / "best", self.best)
        self.checkpoints += 1
        if self.on_snapshot is not None:
            self.on_snapshot(
                checkpoint,
                UniversalCheckpointManifest.model_validate_json(
                    checkpoint.joinpath("manifest.json").read_text(encoding="utf-8")
                ),
            )


def train_universal_game(
    factory: EnvFactory,
    *,
    env_name: str,
    config: UniversalGameConfig,
    output: Path,
    on_snapshot: SnapshotCallback | None = None,
    close_env: bool = True,
) -> UniversalGameRun:
    """Train a seeded DQN on a connected Gymnasium game and publish checkpoints."""

    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    env = _prepare_env(factory, seed=config.seed)
    try:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("prepared environment lost its Discrete action space")
        shape = env.observation_space.shape
        observation_size = int(np.prod(shape if shape is not None else (1,)))
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.30,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [128, 128]},
            seed=config.seed,
            device="cpu",
            verbose=0,
        )
        callback = _CheckpointCallback(
            output=destination,
            env_name=env_name,
            config=config,
            action_count=int(env.action_space.n),
            observation_size=observation_size,
            on_snapshot=on_snapshot,
        )
        model.learn(total_timesteps=config.total_steps, callback=callback, progress_bar=False)
        if callback.latest is None:
            callback._publish()
        assert callback.latest is not None and callback.best is not None
        write_json_atomic(
            destination / "run.json",
            {
                "protocol": "universal-gym-run-v1",
                "env_name": env_name,
                "seed": config.seed,
                "total_steps": config.total_steps,
                "checkpoints": callback.checkpoints,
                "latest": str(callback.latest.relative_to(destination)),
                "best": str(callback.best.relative_to(destination)),
            },
        )
        return UniversalGameRun(
            output=destination,
            latest=destination / "latest",
            best=destination / "best",
            completed_steps=config.total_steps,
            checkpoints=callback.checkpoints,
        )
    finally:
        if close_env:
            env.close()


def _frame_surface(frame: Any, width: int, height: int) -> pygame.Surface | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3, 4) and array.shape[1] > 16:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        low, high = float(np.min(array)), float(np.max(array))
        if high > low:
            array = ((array - low) * 255.0 / (high - low)).astype(np.uint8)
        else:
            array = np.zeros_like(array, dtype=np.uint8)
    surface = pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))
    return pygame.transform.smoothscale(surface, (width, height))


def run_universal_dashboard(
    factory: EnvFactory,
    train: Callable[[SnapshotCallback], UniversalGameRun],
    *,
    env_name: str,
    seed: int = 7,
    max_frames: int | None = None,
    demo_frames: int = 90,
) -> UniversalGameRun:
    """Show the connected game and live learning telemetry while training."""

    if demo_frames < 1:
        raise ValueError("demo_frames must be positive")
    pygame.init()
    updates: deque[tuple[Path, UniversalCheckpointManifest]] = deque()
    lock = threading.Lock()
    result: list[UniversalGameRun] = []
    errors: list[BaseException] = []

    def publish(path: Path, manifest: UniversalCheckpointManifest) -> None:
        with lock:
            updates.append((path, manifest))

    def worker() -> None:
        try:
            result.append(train(publish))
        except BaseException as error:
            errors.append(error)

    env: gym.Env[Any, Any] | None = None
    playback_envs: list[gym.Env[Any, Any]] = []
    model: DQN | None = None
    current: UniversalCheckpointManifest | None = None
    last_action: int | None = None
    last_reward = 0.0
    episodes = 0
    played_frames = 0
    frame_count = 0
    observation: Any = None
    try:
        screen = pygame.display.set_mode((1280, 760))
        pygame.display.set_caption(f"FlyBrain — {env_name} live learning")
        font = pygame.font.Font(None, 28)
        small = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        thread = threading.Thread(target=worker, name="flybrain-universal-training")
        thread.start()
        running = True
        paused = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = not paused

            update: tuple[Path, UniversalCheckpointManifest] | None = None
            with lock:
                if updates and (current is None or frame_count >= demo_frames):
                    update = updates.popleft()
            if update is not None:
                checkpoint, current = update
                env = _prepare_env(factory, seed=seed)
                playback_envs.append(env)
                model = DQN.load(checkpoint / "model.zip", env=env, device="cpu")
                observation, _reset_info = env.reset(seed=seed)
                frame_count = 0

            screen.fill((18, 24, 37))
            status = "training" if thread.is_alive() else "training finished"
            if errors:
                status = f"training failed: {str(errors[0])[:70]}"
            screen.blit(
                font.render(f"FLYBRAIN GAME BRIDGE  |  {env_name}", True, (248, 250, 255)),
                (20, 16),
            )
            screen.blit(
                small.render(
                    f"{status}  |  P pause  |  Q close  |  checkpoints: "
                    f"{current.completed_steps if current else 0}",
                    True,
                    (178, 195, 216),
                ),
                (20, 48),
            )

            if env is not None and model is not None and observation is not None and not paused:
                prediction, _ = model.predict(observation, deterministic=True)
                last_action = int(np.asarray(prediction).item())
                observation, reward, terminated, truncated, _step_info = env.step(last_action)
                last_reward = float(reward)
                episodes += int(terminated or truncated)
                if terminated or truncated:
                    observation, _reset_info = env.reset(seed=seed + episodes)
                frame_count += 1
                played_frames += 1

            if env is not None:
                frame = _frame_surface(env.render(), 880, 600)
                if frame is not None:
                    screen.blit(frame, (20, 92))
                else:
                    screen.blit(
                        small.render(
                            "Connected game has no RGB render() frame",
                            True,
                            (247, 150, 150),
                        ),
                        (35, 120),
                    )

            panel_x = 930
            panel_lines = [
                f"mode: {status}",
                f"step: {current.completed_steps if current else 0}",
                f"action: {last_action if last_action is not None else '-'}",
                f"reward: {last_reward:.3f}",
                f"episodes: {episodes}",
                f"input size: {current.observation_size if current else '-'}",
                f"actions: {current.action_count if current else '-'}",
                "",
                "neural activity",
            ]
            for index, line in enumerate(panel_lines):
                color = (88, 227, 183) if line == "neural activity" else (232, 237, 245)
                screen.blit(small.render(line, True, color), (panel_x, 105 + index * 28))
            values = np.asarray(observation).reshape(-1) if observation is not None else np.zeros(1)
            values = np.abs(values[:32])
            peak = max(1e-6, float(np.max(values)))
            for index, value in enumerate(values):
                bar_height = int(160 * float(value) / peak)
                pygame.draw.rect(
                    screen,
                    (88, 227, 183),
                    pygame.Rect(
                        panel_x + (index % 8) * 38,
                        405 + (index // 8) * 42 - bar_height,
                        24,
                        bar_height,
                    ),
                )
            pygame.display.flip()
            clock.tick(60)
            if errors and not thread.is_alive():
                running = False
            if (
                max_frames is not None
                and not thread.is_alive()
                and not updates
                and frame_count >= demo_frames
                and played_frames >= max_frames
            ):
                running = False
        thread.join()
        if errors:
            raise RuntimeError("universal game training failed") from errors[0]
        if not result:
            raise RuntimeError("universal game training returned no result")
        return result[0]
    finally:
        pygame.quit()
        for playback_env in playback_envs:
            playback_env.close()


def render_connection_state(
    env: gym.Env[Any, Any], observation: Any, *, action: int | None, reward: float
) -> str:
    """Return a concise machine-readable state for external viewers and adapters."""

    values = np.asarray(observation).reshape(-1)
    return json.dumps(
        {
            "game": env.__class__.__name__,
            "action": action,
            "reward": float(reward),
            "observation_shape": list(values.shape),
            "observation_l1": float(np.sum(np.abs(values))),
        },
        sort_keys=True,
    )
