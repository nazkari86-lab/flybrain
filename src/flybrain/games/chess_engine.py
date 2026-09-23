"""Explicit discovery and lifecycle management for an optional Stockfish teacher."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import chess.engine
from pydantic import BaseModel, ConfigDict, Field


class StockfishInfo(BaseModel, frozen=True):
    """Verified identity of one locally executable UCI engine."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    name: str = Field(min_length=1)
    author: str
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_paths(explicit: Path | None) -> tuple[Path, ...]:
    if explicit is not None:
        return (explicit.expanduser(),)
    candidates: list[Path] = []
    discovered = shutil.which("stockfish")
    if discovered is not None:
        candidates.append(Path(discovered))
    candidates.extend((Path("/opt/homebrew/bin/stockfish"), Path("/usr/local/bin/stockfish")))
    return tuple(dict.fromkeys(candidates))


def resolve_stockfish(explicit: Path | None = None) -> StockfishInfo | None:
    """Return probed engine metadata or `None`; never fabricate availability."""

    for candidate in _candidate_paths(explicit):
        resolved = candidate.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            continue
        engine: chess.engine.SimpleEngine | None = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(resolved))
            name = engine.id.get("name", "").strip()
            if not name:
                continue
            return StockfishInfo(
                path=resolved,
                name=name,
                author=engine.id.get("author", "").strip(),
                executable_sha256=_sha256(resolved),
            )
        except (
            FileNotFoundError,
            PermissionError,
            chess.engine.EngineError,
            chess.engine.EngineTerminatedError,
        ):
            continue
        finally:
            if engine is not None:
                engine.quit()
    return None


@contextmanager
def open_stockfish(info: StockfishInfo) -> Iterator[chess.engine.SimpleEngine]:
    """Open exactly the previously verified binary and always terminate it."""

    if _sha256(info.path) != info.executable_sha256:
        raise ValueError("Stockfish executable changed after discovery")
    engine = chess.engine.SimpleEngine.popen_uci(str(info.path))
    try:
        yield engine
    finally:
        engine.quit()
