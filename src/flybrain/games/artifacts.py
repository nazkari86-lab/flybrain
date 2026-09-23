"""Reproducible seed allocation and fail-closed artifact publication."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class SeedSchedule(BaseModel, frozen=True):
    """Named, reproducible, non-overlapping seeds for every data split."""

    model_config = ConfigDict(extra="forbid")

    root_seed: int = Field(ge=0)
    training: tuple[int, ...]
    validation: tuple[int, ...]
    holdout: tuple[int, ...]

    @classmethod
    def build(
        cls,
        root_seed: int,
        *,
        train_count: int,
        validation_count: int,
        holdout_count: int,
    ) -> SeedSchedule:
        """Derive unique positive 31-bit seeds without touching global RNG state."""

        counts = (train_count, validation_count, holdout_count)
        if root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if any(count < 1 for count in counts):
            raise ValueError("every seed split must contain at least one seed")
        total = sum(counts)
        values = random.Random(root_seed).sample(range(1, 2**31), total)
        train_end = train_count
        validation_end = train_end + validation_count
        return cls(
            root_seed=root_seed,
            training=tuple(values[:train_end]),
            validation=tuple(values[train_end:validation_end]),
            holdout=tuple(values[validation_end:]),
        )


class RunManifest(BaseModel, frozen=True):
    """Minimum immutable identity shared by all game-learning runs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    game: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    root_seed: int = Field(ge=0)
    seeds: SeedSchedule
    config: dict[str, JsonValue]
    dependencies: dict[str, str]


def _serialize_json(payload: Mapping[str, JsonValue]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> Path:
    """Publish one JSON file atomically while refusing an existing destination."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / target.name
        with stage.open("xb") as stream:
            stream.write(_serialize_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(stage, target)
        _sync_directory(target.parent)
    return target


def publish_directory_atomic(target: Path, writer: Callable[[Path], None]) -> Path:
    """Build a complete sibling directory and publish it in one rename."""

    destination = target.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        writer(stage)
        for item in sorted(stage.rglob("*")):
            if item.is_file():
                with item.open("rb") as stream:
                    os.fsync(stream.fileno())
        _sync_directory(stage)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(stage, destination)
        _sync_directory(destination.parent)
    return destination
