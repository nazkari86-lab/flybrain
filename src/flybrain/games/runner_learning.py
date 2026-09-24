"""Persistent DQN training and immutable evaluation for the procedural runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import DQN

from flybrain.games.artifacts import SeedSchedule, publish_directory_atomic, write_json_atomic
from flybrain.games.contracts import GameMode
from flybrain.games.runner_env import RunnerConfig, RunnerEnv


class RunnerTrainingConfig(BaseModel, frozen=True):
    """Complete bounded configuration for one resumable DQN run."""

    model_config = ConfigDict(extra="forbid")

    total_steps: int = Field(default=100_000, ge=1, le=100_000_000)
    checkpoint_every: int = Field(default=10_000, ge=1, le=10_000_000)
    seed: int = Field(default=7, ge=0)
    training_seed_count: int = Field(default=32, ge=1, le=100_000)
    validation_seed_count: int = Field(default=4, ge=1, le=10_000)
    holdout_seed_count: int = Field(default=8, ge=1, le=10_000)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)


class RunnerCheckpointManifest(BaseModel, frozen=True):
    """Identity and complete continuation metadata for one checkpoint."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["runner-dqn-v1"] = "runner-dqn-v1"
    algorithm: Literal["stable-baselines3-dqn"] = "stable-baselines3-dqn"
    completed_steps: int = Field(ge=0)
    training_seed_index: int = Field(ge=0)
    config: RunnerTrainingConfig
    seeds: SeedSchedule
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_mean_distance: float = Field(ge=0.0, le=1.0)
    dependencies: dict[str, str]


class RunnerEpisodeResult(BaseModel, frozen=True):
    """Metrics from one frozen-policy episode."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    completed: bool
    collision: bool
    normalized_distance: float = Field(ge=0.0, le=1.0)
    episode_return: float
    steps: int = Field(ge=1)
    jump_actions: int = Field(ge=0)


class RunnerEvaluation(BaseModel, frozen=True):
    """Immutable multi-seed runner evaluation report."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["runner-evaluation-v1"] = "runner-evaluation-v1"
    episodes: int = Field(ge=1)
    completions: int = Field(ge=0)
    collisions: int = Field(ge=0)
    completion_rate: float = Field(ge=0.0, le=1.0)
    mean_normalized_distance: float = Field(ge=0.0, le=1.0)
    confidence_low: float = Field(ge=0.0, le=1.0)
    confidence_high: float = Field(ge=0.0, le=1.0)
    mean_return: float
    actions_per_step: float = Field(ge=0.0, le=1.0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_mutated: bool
    per_seed: tuple[RunnerEpisodeResult, ...]


class RunnerRun(BaseModel, frozen=True):
    """Paths and identity returned after a successful training segment."""

    model_config = ConfigDict(extra="forbid")

    output: Path
    latest: Path
    best: Path
    completed_steps: int = Field(ge=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _SeedCyclingRunnerEnv(RunnerEnv):
    def __init__(
        self,
        config: RunnerConfig,
        seeds: tuple[int, ...],
        *,
        mode: GameMode,
        start_index: int = 0,
    ) -> None:
        super().__init__(config)
        self._episode_seeds = seeds
        self._episode_mode = mode
        self.seed_index = start_index

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = self._episode_seeds[self.seed_index % len(self._episode_seeds)]
            self.seed_index += 1
        merged = dict(options or {})
        merged["mode"] = self._episode_mode
        return super().reset(seed=seed, options=merged)


def _dependencies() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("gymnasium", "numpy", "stable-baselines3", "torch")
    }


def _hash_file(path: Path) -> str:
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
        digest.update(bytes.fromhex(_hash_file(item)))
    return digest.hexdigest()


def _schedule(config: RunnerTrainingConfig) -> SeedSchedule:
    return SeedSchedule.build(
        config.seed,
        train_count=config.training_seed_count,
        validation_count=config.validation_seed_count,
        holdout_count=config.holdout_seed_count,
    )


def _new_model(env: RunnerEnv, config: RunnerTrainingConfig) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=64,
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


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=link.parent) as temporary:
        staged = Path(temporary) / link.name
        staged.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        os.replace(staged, link)


def _checkpoint_manifest(path: Path) -> RunnerCheckpointManifest:
    return RunnerCheckpointManifest.model_validate_json(
        path.resolve().joinpath("manifest.json").read_text()
    )


def _compatible_resume(old: RunnerTrainingConfig, new: RunnerTrainingConfig) -> bool:
    excluded = {"total_steps", "checkpoint_every"}
    return old.model_dump(exclude=excluded) == new.model_dump(exclude=excluded)


def _run_episode(model: DQN, config: RunnerConfig, seed: int) -> RunnerEpisodeResult:
    env = RunnerEnv(config)
    observation, info = env.reset(seed=seed, options={"mode": "holdout"})
    total_reward = 0.0
    jumps = 0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        prediction, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(prediction).item())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        jumps += int(action == 1)
        steps += 1
    return RunnerEpisodeResult(
        seed=seed,
        completed=bool(info["completed"]),
        collision=bool(info["collision"]),
        normalized_distance=float(info["normalized_distance"]),
        episode_return=total_reward,
        steps=steps,
        jump_actions=jumps,
    )


def _mean_distance(model: DQN, config: RunnerConfig, seeds: tuple[int, ...]) -> float:
    values = [_run_episode(model, config, seed).normalized_distance for seed in seeds]
    return float(np.mean(values))


def _run_random_episode(
    config: RunnerConfig,
    seed: int,
    generator: np.random.Generator,
) -> RunnerEpisodeResult:
    env = RunnerEnv(config)
    _, info = env.reset(seed=seed, options={"mode": "holdout"})
    total_reward = 0.0
    jumps = 0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(generator.integers(0, 2))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        jumps += int(action == 1)
        steps += 1
    return RunnerEpisodeResult(
        seed=seed,
        completed=bool(info["completed"]),
        collision=bool(info["collision"]),
        normalized_distance=float(info["normalized_distance"]),
        episode_return=total_reward,
        steps=steps,
        jump_actions=jumps,
    )


def _evaluation_from_episodes(
    episodes: tuple[RunnerEpisodeResult, ...],
    *,
    model_sha256: str,
    training_mutated: bool,
) -> RunnerEvaluation:
    distances = np.asarray([item.normalized_distance for item in episodes], dtype=np.float64)
    generator = np.random.default_rng(0)
    samples = np.asarray(
        [float(np.mean(generator.choice(distances, size=len(distances)))) for _ in range(500)]
    )
    total_steps = sum(item.steps for item in episodes)
    return RunnerEvaluation(
        episodes=len(episodes),
        completions=sum(item.completed for item in episodes),
        collisions=sum(item.collision for item in episodes),
        completion_rate=sum(item.completed for item in episodes) / len(episodes),
        mean_normalized_distance=float(np.mean(distances)),
        confidence_low=float(np.quantile(samples, 0.025)),
        confidence_high=float(np.quantile(samples, 0.975)),
        mean_return=float(np.mean([item.episode_return for item in episodes])),
        actions_per_step=sum(item.jump_actions for item in episodes) / total_steps,
        model_sha256=model_sha256,
        training_mutated=training_mutated,
        per_seed=episodes,
    )


def _policy_sha256(model: DQN) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(
    target: Path,
    model: DQN,
    env: _SeedCyclingRunnerEnv,
    config: RunnerTrainingConfig,
    seeds: SeedSchedule,
    validation_mean_distance: float,
) -> RunnerCheckpointManifest:
    holder: list[RunnerCheckpointManifest] = []

    def write(stage: Path) -> None:
        model.save(stage / "model")
        model.save_replay_buffer(stage / "replay.pkl")
        model_hash = _hash_file(stage / "model.zip")
        manifest = RunnerCheckpointManifest(
            completed_steps=int(model.num_timesteps),
            training_seed_index=env.seed_index,
            config=config,
            seeds=seeds,
            model_sha256=model_hash,
            validation_mean_distance=validation_mean_distance,
            dependencies=_dependencies(),
        )
        write_json_atomic(
            stage / "manifest.json",
            json.loads(manifest.model_dump_json()),
        )
        holder.append(manifest)

    publish_directory_atomic(target, write)
    return holder[0]


def load_runner_model(checkpoint: Path, env: RunnerEnv | None = None) -> DQN:
    """Load one frozen DQN policy from a validated runner checkpoint."""

    resolved = checkpoint.resolve()
    manifest = _checkpoint_manifest(resolved)
    model_path = resolved / "model.zip"
    if _hash_file(model_path) != manifest.model_sha256:
        raise ValueError("runner model hash does not match checkpoint manifest")
    return DQN.load(model_path, env=env, device="cpu")


def train_runner(
    config: RunnerTrainingConfig,
    output: Path,
    resume: Path | None = None,
    on_checkpoint: Callable[[Path, RunnerCheckpointManifest], None] | None = None,
) -> RunnerRun:
    """Train to an absolute timestep target and publish immutable checkpoints."""

    root = output.resolve()
    seeds = _schedule(config)
    start_index = 0
    if resume is None:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        write_json_atomic(
            root / "run.json",
            {
                "config": json.loads(config.model_dump_json()),
                "protocol": "runner-run-v1",
                "seeds": json.loads(seeds.model_dump_json()),
            },
        )
        env = _SeedCyclingRunnerEnv(config.runner, seeds.training, mode="training")
        model = _new_model(env, config)
    else:
        if not root.is_dir():
            raise FileNotFoundError(root)
        resume_path = resume.resolve()
        old = _checkpoint_manifest(resume_path)
        if not _compatible_resume(old.config, config):
            raise ValueError("resume checkpoint is incompatible with requested runner config")
        if old.seeds != seeds:
            raise ValueError("resume seed schedule does not match requested runner config")
        if config.total_steps <= old.completed_steps:
            raise ValueError("total_steps must exceed the resumed checkpoint")
        start_index = old.training_seed_index
        env = _SeedCyclingRunnerEnv(
            config.runner,
            seeds.training,
            mode="training",
            start_index=start_index,
        )
        model = load_runner_model(resume_path, env)
        model.load_replay_buffer(resume_path / "replay.pkl")

    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    best_score = -1.0
    if root.joinpath("best").exists():
        best_score = _checkpoint_manifest(root / "best").validation_mean_distance

    while int(model.num_timesteps) < config.total_steps:
        remaining = config.total_steps - int(model.num_timesteps)
        chunk = min(config.checkpoint_every, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        validation_score = _mean_distance(model, config.runner, seeds.validation)
        step = int(model.num_timesteps)
        checkpoint = checkpoints / f"step-{step:012d}"
        manifest = _save_checkpoint(
            checkpoint,
            model,
            env,
            config,
            seeds,
            validation_score,
        )
        _replace_symlink(root / "latest", checkpoint)
        if validation_score >= best_score:
            _replace_symlink(root / "best", checkpoint)
            best_score = validation_score
        with root.joinpath("metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "completed_steps": step,
                        "model_sha256": manifest.model_sha256,
                        "validation_mean_distance": validation_score,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if on_checkpoint is not None:
            on_checkpoint(checkpoint, manifest)

    latest = root / "latest"
    latest_manifest = _checkpoint_manifest(latest)
    return RunnerRun(
        output=root,
        latest=latest,
        best=root / "best",
        completed_steps=latest_manifest.completed_steps,
        model_sha256=latest_manifest.model_sha256,
    )


def evaluate_runner(checkpoint: Path, seeds: tuple[int, ...]) -> RunnerEvaluation:
    """Evaluate a separately loaded deterministic policy without any training writes."""

    if not seeds:
        raise ValueError("runner evaluation requires at least one seed")
    resolved = checkpoint.resolve()
    before = _hash_tree(resolved)
    manifest = _checkpoint_manifest(resolved)
    model = load_runner_model(resolved)
    episodes = tuple(_run_episode(model, manifest.config.runner, seed) for seed in seeds)
    after = _hash_tree(resolved)
    return _evaluation_from_episodes(
        episodes,
        model_sha256=manifest.model_sha256,
        training_mutated=before != after,
    )


def evaluate_random_runner(
    config: RunnerConfig,
    *,
    seeds: tuple[int, ...],
    policy_seed: int = 0,
) -> RunnerEvaluation:
    """Evaluate a fixed seeded random-action baseline."""

    if not seeds:
        raise ValueError("runner evaluation requires at least one seed")
    generator = np.random.default_rng(policy_seed)
    episodes = tuple(_run_random_episode(config, seed, generator) for seed in seeds)
    identity = hashlib.sha256(
        f"random-runner-v1:{policy_seed}:{config.model_dump_json()}".encode()
    ).hexdigest()
    return _evaluation_from_episodes(
        episodes,
        model_sha256=identity,
        training_mutated=False,
    )


def evaluate_untrained_runner(
    config: RunnerConfig,
    *,
    seeds: tuple[int, ...],
    policy_seed: int = 0,
) -> RunnerEvaluation:
    """Evaluate the exact DQN architecture before any gradient update."""

    if not seeds:
        raise ValueError("runner evaluation requires at least one seed")
    schedule = tuple(seeds)
    env = _SeedCyclingRunnerEnv(config, schedule, mode="holdout")
    model = _new_model(
        env,
        RunnerTrainingConfig(
            total_steps=1,
            checkpoint_every=1,
            seed=policy_seed,
            runner=config,
        ),
    )
    episodes = tuple(_run_episode(model, config, seed) for seed in seeds)
    return _evaluation_from_episodes(
        episodes,
        model_sha256=_policy_sha256(model),
        training_mutated=False,
    )
