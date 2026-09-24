"""Immutable, serializable continuation of fast plasticity and slow DA/NO memory."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.mushroom_body_learning import MushroomBodyLearning
from flybrain.slow_memory import SlowMemoryState


class SlowMemorySnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    nitric_oxide_competent: tuple[bool, ...]
    dopamine_immediate: tuple[float, ...]
    nitric_oxide_immediate: tuple[float, ...]
    dopamine_effect: tuple[float, ...]
    nitric_oxide_effect: tuple[float, ...]

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        for values in (self.dopamine_immediate, self.nitric_oxide_immediate,
                       self.dopamine_effect, self.nitric_oxide_effect):
            if len(values) != len(self.nitric_oxide_competent):
                raise ValueError("slow memory shape mismatch")
            if any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in values):
                raise ValueError("slow memory values must be finite and in [0, 1]")
        return self

    @classmethod
    def capture(cls, state: SlowMemoryState) -> SlowMemorySnapshot:
        return cls(**{
            name: tuple(getattr(state, name).tolist()) for name in cls.model_fields
        })

    def restore(self, state: SlowMemoryState) -> None:
        if tuple(state.nitric_oxide_competent.tolist()) != self.nitric_oxide_competent:
            raise ValueError("slow memory compartment identity mismatch")
        for name in type(self).model_fields:
            if name != "nitric_oxide_competent":
                getattr(state, name)[:] = getattr(self, name)


class AutonomousLearningMemory(BaseModel, frozen=True):
    """Learning state only; the body and fast neural voltages reset per episode.

    The context digest binds anatomy, edge ordering, DAN routing and parameters.
    Effective weights are kept separately from fast weights so slow modulation
    is neither discarded nor multiplied into the fast base a second time.
    """

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["autonomous-learning-memory-v1"] = "autonomous-learning-memory-v1"
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_multipliers: tuple[float, ...]
    effective_multipliers: tuple[float, ...]
    eligibility: tuple[float, ...]
    dopamine_trace: tuple[float, ...]
    slow: SlowMemorySnapshot | None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        n = len(self.base_multipliers)
        if len(self.effective_multipliers) != n or len(self.eligibility) != n:
            raise ValueError("learning memory edge shapes differ")
        if self.slow is not None and len(self.slow.dopamine_effect) != n:
            raise ValueError("learning memory slow-state shape differs")
        for values in (self.base_multipliers, self.effective_multipliers,
                       self.eligibility, self.dopamine_trace):
            if any(not math.isfinite(x) or x < 0.0 for x in values):
                raise ValueError("learning memory values must be finite and non-negative")
        if any(x > 2.0 for x in (*self.base_multipliers, *self.effective_multipliers)):
            raise ValueError("learning memory weights exceed supported bounds")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @classmethod
    def capture(
        cls, learner: MushroomBodyLearning, slow: SlowMemoryState | None,
        effective: NDArray[np.float32], context: str,
    ) -> AutonomousLearningMemory:
        return cls(
            context_sha256=context,
            base_multipliers=tuple(float(x) for x in learner.overlay.multipliers),
            effective_multipliers=tuple(float(x) for x in effective),
            eligibility=tuple(float(x) for x in learner.eligibility),
            dopamine_trace=tuple(float(x) for x in learner.dopamine_trace),
            slow=SlowMemorySnapshot.capture(slow) if slow is not None else None,
        )

    def restore(
        self, learner: MushroomBodyLearning, slow: SlowMemoryState | None,
        effective: NDArray[np.float32], context: str,
    ) -> None:
        if context != self.context_sha256:
            raise ValueError("learning memory context identity mismatch")
        if tuple(float(x) for x in effective) != self.effective_multipliers:
            raise ValueError("learning memory effective weights mismatch")
        if len(self.eligibility) != learner.eligibility.size:
            raise ValueError("learning memory eligibility shape mismatch")
        if len(self.dopamine_trace) != learner.dopamine_trace.size:
            raise ValueError("learning memory dopamine shape mismatch")
        if (slow is None) != (self.slow is None):
            raise ValueError("learning memory slow-state identity mismatch")
        if slow is not None and self.slow is not None:
            self.slow.restore(slow)
        learner.overlay.set_multipliers(
            np.asarray(self.base_multipliers, dtype=np.float32), minimum=0.0, maximum=2.0,
        )
        learner.eligibility[:] = self.eligibility
        learner.dopamine_trace[:] = self.dopamine_trace


class LearningMemoryCheckpoint(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    memory: AutonomousLearningMemory
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if self.digest != self.memory.digest:
            raise ValueError("learning memory checkpoint digest mismatch")
        return self


def save_learning_memory(path: Path, memory: AutonomousLearningMemory) -> None:
    """Publish a complete state atomically without replacing any existing file."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = LearningMemoryCheckpoint(memory=memory, digest=memory.digest)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(checkpoint.model_dump_json())
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_learning_memory(path: Path) -> AutonomousLearningMemory:
    """Verify serialized content; episode restore separately checks context and weights."""
    return LearningMemoryCheckpoint.model_validate_json(path.read_text()).memory
