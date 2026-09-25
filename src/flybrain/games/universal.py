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
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from flybrain.games.artifacts import (
    SeedSchedule,
    publish_directory_atomic,
    write_json_atomic,
)

EnvFactory = Callable[[], gym.Env[Any, Any]]
SnapshotCallback = Callable[[Path, "UniversalCheckpointManifest"], None]


class UniversalGameConfig(BaseModel, frozen=True):
    """Complete reproducible configuration for one generic discrete-action game."""

    model_config = ConfigDict(extra="forbid")

    total_steps: int = Field(default=100_000, ge=1, le=100_000_000)
    checkpoint_every: int = Field(default=10_000, ge=1, le=10_000_000)
    seed: int = Field(default=7, ge=0)
    training_seed_count: int = Field(default=32, ge=1, le=100_000)
    validation_seed_count: int = Field(default=4, ge=1, le=10_000)
    holdout_seed_count: int = Field(default=8, ge=1, le=10_000)
    buffer_size: int = Field(default=50_000, ge=128, le=5_000_000)
    learning_starts: int = Field(default=500, ge=0, le=5_000_000)
    batch_size: int = Field(default=64, ge=1, le=4096)
    evaluation_max_steps: int = Field(default=10_000, ge=1, le=10_000_000)


class UniversalCheckpointManifest(BaseModel, frozen=True):
    """Identity and complete continuation metadata for one generic-game checkpoint."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["universal-gym-dqn-v2"] = "universal-gym-dqn-v2"
    algorithm: Literal["stable-baselines3-dqn"] = "stable-baselines3-dqn"
    env_name: str = Field(min_length=1)
    completed_steps: int = Field(ge=0)
    training_seed_index: int = Field(ge=0)
    config: UniversalGameConfig
    seeds: SeedSchedule
    episodes: int = Field(ge=0)
    action_count: int = Field(ge=2)
    observation_size: int = Field(ge=1)
    space_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recent_mean_reward: float
    validation_mean_reward: float
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: dict[str, str]


class UniversalGameRun(BaseModel, frozen=True):
    """Paths and progress returned by a completed connection run."""

    model_config = ConfigDict(extra="forbid")

    output: Path
    latest: Path
    best: Path
    completed_steps: int = Field(ge=1)
    checkpoints: int = Field(ge=1)


class UniversalEpisodeResult(BaseModel, frozen=True):
    """One frozen-policy episode from a generic connected environment."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    episode_return: float
    steps: int = Field(ge=1)
    terminated: bool
    truncated: bool


class UniversalEvaluation(BaseModel, frozen=True):
    """Immutable holdout evaluation for a generic connected environment."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["universal-evaluation-v1"] = "universal-evaluation-v1"
    episodes: int = Field(ge=1)
    mean_return: float
    confidence_low: float
    confidence_high: float
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_mutated: bool
    per_seed: tuple[UniversalEpisodeResult, ...]


@dataclass(frozen=True)
class _FactorySpec:
    factory: EnvFactory
    name: str
    visual_factory: EnvFactory


def load_factory(spec: str) -> _FactorySpec:
    """Load a local ``module:callable`` environment factory."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use the module:callable form")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"factory is not callable: {spec}")
    return _FactorySpec(factory=factory, name=spec, visual_factory=factory)


def env_id_factory(env_id: str) -> _FactorySpec:
    """Create a connector for any registered Gymnasium environment."""

    if not env_id.strip():
        raise ValueError("env_id must not be empty")

    def factory() -> gym.Env[Any, Any]:
        return gym.make(env_id)

    def visual_factory() -> gym.Env[Any, Any]:
        try:
            return gym.make(env_id, render_mode="rgb_array")
        except TypeError:
            return gym.make(env_id)

    return _FactorySpec(factory=factory, name=env_id, visual_factory=visual_factory)


class _SeedCyclingEnv(gym.Wrapper[Any, Any, Any, Any]):
    """Feed deterministic, non-overlapping episode seeds into a Gymnasium env."""

    def __init__(self, env: gym.Env[Any, Any], seeds: tuple[int, ...], start_index: int = 0):
        super().__init__(env)
        if not seeds:
            raise ValueError("training seed schedule must not be empty")
        self.episode_seeds = seeds
        self.seed_index = start_index

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is None:
            seed = self.episode_seeds[self.seed_index % len(self.episode_seeds)]
            self.seed_index += 1
        return self.env.reset(seed=seed, options=options)


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


def _hash_tree(path: Path) -> str:
    root = path.resolve()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _space_sha256(env: gym.Env[Any, Any]) -> str:
    payload = {
        "action_space": repr(env.action_space),
        "observation_space": repr(env.observation_space),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _schedule(config: UniversalGameConfig) -> SeedSchedule:
    return SeedSchedule.build(
        config.seed,
        train_count=config.training_seed_count,
        validation_count=config.validation_seed_count,
        holdout_count=config.holdout_seed_count,
    )


def _checkpoint_manifest(path: Path) -> UniversalCheckpointManifest:
    return UniversalCheckpointManifest.model_validate_json(
        path.resolve().joinpath("manifest.json").read_text(encoding="utf-8")
    )


def _compatible_resume(old: UniversalGameConfig, new: UniversalGameConfig) -> bool:
    excluded = {"total_steps", "checkpoint_every"}
    return old.model_dump(exclude=excluded) == new.model_dump(exclude=excluded)


def _run_episode(
    model: DQN,
    factory: EnvFactory,
    *,
    seed: int,
    max_steps: int,
) -> UniversalEpisodeResult:
    env = _prepare_env(factory)
    try:
        observation, _reset_info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and steps < max_steps:
            prediction, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(prediction).item())
            observation, reward, terminated, truncated, _step_info = env.step(action)
            total_reward += float(reward)
            steps += 1
        if not terminated and not truncated and steps >= max_steps:
            truncated = True
        return UniversalEpisodeResult(
            seed=seed,
            episode_return=total_reward,
            steps=steps,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
    finally:
        env.close()


def _evaluation_from_episodes(
    episodes: tuple[UniversalEpisodeResult, ...],
    *,
    model_sha256: str,
    training_mutated: bool,
) -> UniversalEvaluation:
    values = np.asarray([episode.episode_return for episode in episodes], dtype=np.float64)
    generator = np.random.default_rng(0)
    samples = np.asarray(
        [float(np.mean(generator.choice(values, size=len(values)))) for _ in range(500)]
    )
    return UniversalEvaluation(
        episodes=len(episodes),
        mean_return=float(np.mean(values)),
        confidence_low=float(np.quantile(samples, 0.025)),
        confidence_high=float(np.quantile(samples, 0.975)),
        model_sha256=model_sha256,
        training_mutated=training_mutated,
        per_seed=episodes,
    )


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
        space_sha256: str,
        seeds: SeedSchedule,
        seed_tracker: _SeedCyclingEnv,
        validation_factory: EnvFactory,
        validation_seeds: tuple[int, ...],
        on_snapshot: SnapshotCallback | None = None,
        initial_best_score: float = float("-inf"),
    ) -> None:
        super().__init__(verbose=0)
        self.output = output
        self.env_name = env_name
        self.config = config
        self.action_count = action_count
        self.observation_size = observation_size
        self.space_sha256 = space_sha256
        self.seeds = seeds
        self.seed_tracker = seed_tracker
        self.validation_factory = validation_factory
        self.validation_seeds = validation_seeds
        self.on_snapshot = on_snapshot
        self.checkpoints = 0
        self.episodes = 0
        self.rewards: deque[float] = deque(maxlen=100)
        self.best_score = initial_best_score
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
        model = cast(DQN, self.model)
        step = int(self.num_timesteps)
        destination = self.output / "checkpoints" / f"step-{step:012d}"
        training_score = float(np.mean(self.rewards)) if self.rewards else 0.0
        validation_episodes = tuple(
            _run_episode(
                model,
                self.validation_factory,
                seed=seed,
                max_steps=self.config.evaluation_max_steps,
            )
            for seed in self.validation_seeds
        )
        validation_score = float(
            np.mean([episode.episode_return for episode in validation_episodes])
        )

        def writer(stage: Path) -> None:
            model.save(stage / "model")
            model.save_replay_buffer(stage / "replay.pkl")
            model_path = stage / "model.zip"
            replay_path = stage / "replay.pkl"
            manifest = UniversalCheckpointManifest(
                env_name=self.env_name,
                completed_steps=step,
                training_seed_index=self.seed_tracker.seed_index,
                config=self.config,
                seeds=self.seeds,
                episodes=self.episodes,
                action_count=self.action_count,
                observation_size=self.observation_size,
                space_sha256=self.space_sha256,
                recent_mean_reward=training_score,
                validation_mean_reward=validation_score,
                model_sha256=_sha256(model_path),
                replay_sha256=_sha256(replay_path),
                dependencies=_dependencies(),
            )
            stage.joinpath("manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

        checkpoint = publish_directory_atomic(destination, writer)
        self.latest = checkpoint
        if self.best is None or validation_score >= self.best_score:
            self.best_score = validation_score
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


def load_universal_model(checkpoint: Path, env: gym.Env[Any, Any] | None = None) -> DQN:
    """Load a validated generic checkpoint without changing it."""

    resolved = checkpoint.resolve()
    manifest = _checkpoint_manifest(resolved)
    model_path = resolved / "model.zip"
    replay_path = resolved / "replay.pkl"
    if _sha256(model_path) != manifest.model_sha256:
        raise ValueError("universal model hash does not match checkpoint manifest")
    if _sha256(replay_path) != manifest.replay_sha256:
        raise ValueError("universal replay hash does not match checkpoint manifest")
    if env is not None and _space_sha256(env) != manifest.space_sha256:
        raise ValueError("environment spaces do not match universal checkpoint")
    return DQN.load(model_path, env=env, device="cpu")


def train_universal_game(
    factory: EnvFactory,
    *,
    env_name: str,
    config: UniversalGameConfig,
    output: Path,
    resume: Path | None = None,
    on_snapshot: SnapshotCallback | None = None,
    close_env: bool = True,
) -> UniversalGameRun:
    """Train or resume a seeded DQN and publish complete generic-game checkpoints."""

    destination = output.resolve()
    seeds = _schedule(config)
    if resume is None:
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        write_json_atomic(
            destination / "run.json",
            {
                "protocol": "universal-gym-run-v2",
                "env_name": env_name,
                "config": json.loads(config.model_dump_json()),
                "seeds": json.loads(seeds.model_dump_json()),
            },
        )
        start_index = 0
        initial_best_score = float("-inf")
        prepared = _prepare_env(factory)
        training_env = _SeedCyclingEnv(prepared, seeds.training)
        model = DQN(
            "MlpPolicy",
            training_env,
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
    else:
        if not destination.is_dir():
            raise FileNotFoundError(destination)
        resume_path = resume.resolve()
        old = _checkpoint_manifest(resume_path)
        if not _compatible_resume(old.config, config):
            raise ValueError("resume checkpoint is incompatible with requested universal config")
        if old.seeds != seeds:
            raise ValueError("resume seed schedule does not match requested universal config")
        if config.total_steps <= old.completed_steps:
            raise ValueError("total_steps must exceed the resumed checkpoint")
        prepared = _prepare_env(factory)
        if _space_sha256(prepared) != old.space_sha256:
            prepared.close()
            raise ValueError("environment spaces do not match resumed universal checkpoint")
        start_index = old.training_seed_index
        training_env = _SeedCyclingEnv(prepared, seeds.training, start_index=start_index)
        model = load_universal_model(resume_path, env=training_env)
        model.load_replay_buffer(resume_path / "replay.pkl")
        best_path = destination / "best"
        initial_best_score = (
            _checkpoint_manifest(best_path).validation_mean_reward
            if best_path.exists()
            else float("-inf")
        )

    try:
        if not isinstance(training_env.action_space, gym.spaces.Discrete):
            raise ValueError("prepared environment lost its Discrete action space")
        shape = training_env.observation_space.shape
        observation_size = int(np.prod(shape if shape is not None else (1,)))
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        callback = _CheckpointCallback(
            output=destination,
            env_name=env_name,
            config=config,
            action_count=int(training_env.action_space.n),
            observation_size=observation_size,
            space_sha256=_space_sha256(prepared),
            seeds=seeds,
            seed_tracker=training_env,
            validation_factory=factory,
            validation_seeds=seeds.validation,
            on_snapshot=on_snapshot,
            initial_best_score=initial_best_score,
        )
        while int(model.num_timesteps) < config.total_steps:
            remaining = config.total_steps - int(model.num_timesteps)
            chunk = min(config.checkpoint_every, remaining)
            model.learn(total_timesteps=chunk, callback=callback, reset_num_timesteps=False)
        if callback.latest is None:
            callback._publish()
        assert callback.latest is not None and callback.best is not None
        return UniversalGameRun(
            output=destination,
            latest=destination / "latest",
            best=destination / "best",
            completed_steps=int(callback.latest.name.split("-")[-1]),
            checkpoints=callback.checkpoints,
        )
    finally:
        if close_env:
            training_env.close()


def evaluate_universal_game(
    checkpoint: Path,
    factory: EnvFactory,
    *,
    seeds: tuple[int, ...] | None = None,
) -> UniversalEvaluation:
    """Evaluate a frozen universal checkpoint on its declared holdout seeds."""

    resolved = checkpoint.resolve()
    before = _hash_tree(resolved)
    manifest = _checkpoint_manifest(resolved)
    evaluation_seeds = manifest.seeds.holdout if seeds is None else seeds
    if not evaluation_seeds:
        raise ValueError("universal evaluation requires at least one seed")
    env = _prepare_env(factory)
    try:
        model = load_universal_model(resolved, env=env)
    finally:
        env.close()
    episodes = tuple(
        _run_episode(
            model,
            factory,
            seed=seed,
            max_steps=manifest.config.evaluation_max_steps,
        )
        for seed in evaluation_seeds
    )
    after = _hash_tree(resolved)
    return _evaluation_from_episodes(
        episodes,
        model_sha256=manifest.model_sha256,
        training_mutated=before != after,
    )


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
            status = "training + checkpoint playback" if thread.is_alive() else "training finished"
            if errors:
                status = f"training failed: {str(errors[0])[:70]}"
            screen.blit(
                font.render(f"FLYBRAIN GAME BRIDGE  |  {env_name}", True, (248, 250, 255)),
                (20, 16),
            )
            screen.blit(
                small.render(
                    f"{status}  |  P pause  |  Q close  |  checkpoint: "
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
                "mode: checkpoint playback",
                f"checkpoint step: {current.completed_steps if current else 0}",
                f"action: {last_action if last_action is not None else '-'}",
                f"reward: {last_reward:.3f}",
                f"episodes: {episodes}",
                f"input size: {current.observation_size if current else '-'}",
                f"actions: {current.action_count if current else '-'}",
                "",
                "observation activity",
            ]
            for index, line in enumerate(panel_lines):
                color = (88, 227, 183) if line == "observation activity" else (232, 237, 245)
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
