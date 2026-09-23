"""Small shared contracts for independently implemented game learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

GameMode = Literal["training", "validation", "holdout", "play"]
InfoValue = int | float | str | bool | None


@dataclass(frozen=True)
class StepResult[ObservationT]:
    """One environment transition with Gymnasium-compatible termination flags."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, InfoValue]


class GameAdapter[ObservationT, ActionT](Protocol):
    """Minimal game boundary shared by reactive and planning agents."""

    def reset(self, *, seed: int, mode: GameMode) -> ObservationT:
        """Start one deterministic episode."""

        ...

    def legal_actions(self) -> tuple[ActionT, ...]:
        """Return only actions accepted in the current state."""

        ...

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        """Apply one legal action and return the resulting transition."""

        ...
