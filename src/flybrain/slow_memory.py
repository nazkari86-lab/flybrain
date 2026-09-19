"""Evidence-bounded dopamine and nitric-oxide memory dynamics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray


@dataclass(frozen=True)
class NitricOxideDANs:
    """MaleCNS DAN identities linked to nitric-oxide signalling in Aso 2019."""

    ppl101_ids: tuple[int, ...]
    pam01_ids: tuple[int, ...]
    evidence_doi: str = "10.7554/eLife.49257"

    @property
    def all_ids(self) -> tuple[int, ...]:
        """Return all evidence-supported IDs in deterministic order."""

        return tuple(sorted((*self.ppl101_ids, *self.pam01_ids)))


def resolve_nitric_oxide_dans(snapshot: Path) -> NitricOxideDANs:
    """Resolve only the MaleCNS PPL101 and PAM01 DAN types supported by Aso 2019."""

    source = snapshot / "source-annotations.parquet"
    table = pq.read_table(source, columns=["bodyId", "class", "type"])
    records: dict[int, tuple[str, str]] = {}
    for body, neuron_class, neuron_type in zip(
        table.column("bodyId").to_pylist(),
        table.column("class").to_pylist(),
        table.column("type").to_pylist(),
        strict=True,
    ):
        if body is None or neuron_class is None or neuron_type is None:
            raise ValueError("source annotations must not contain null identities")
        body_id = int(body)
        if body_id <= 0:
            raise ValueError("source annotation body IDs must be positive")
        if body_id in records:
            raise ValueError(f"source annotations contain duplicate body ID {body_id}")
        records[body_id] = (str(neuron_class), str(neuron_type))

    ppl101_ids = tuple(
        sorted(
            body_id
            for body_id, identity in records.items()
            if identity == ("DAN", "PPL101")
        )
    )
    pam01_ids = tuple(
        sorted(
            body_id
            for body_id, identity in records.items()
            if identity == ("DAN", "PAM01")
        )
    )
    if not ppl101_ids or not pam01_ids:
        raise ValueError("source annotations require DAN types PPL101 and PAM01")
    return NitricOxideDANs(ppl101_ids=ppl101_ids, pam01_ids=pam01_ids)


@dataclass(frozen=True)
class SlowMemoryParameters:
    """Behavioral-model rates and time constants reported by Aso et al. 2019."""

    dopamine_induction_per_minute: float = 4.3
    nitric_oxide_induction_per_minute: float = 0.96
    dopamine_effect_tau_seconds: float = 30.0
    nitric_oxide_effect_tau_seconds: float = 600.0
    dopamine_unpaired_decay_per_minute: float = 0.26
    nitric_oxide_unpaired_decay_per_minute: float = 0.16

    def validate(self) -> None:
        """Reject values outside the positive finite model domain."""

        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("slow-memory parameters must be finite")
        if not all(value > 0 for value in values.values()):
            raise ValueError("slow-memory parameters must be positive")


@dataclass
class SlowMemoryState:
    """Immediate induction states and their distinct observable effect traces."""

    nitric_oxide_competent: NDArray[np.bool_]
    dopamine_immediate: NDArray[np.float64]
    nitric_oxide_immediate: NDArray[np.float64]
    dopamine_effect: NDArray[np.float64]
    nitric_oxide_effect: NDArray[np.float64]

    @classmethod
    def initial(
        cls,
        *,
        nitric_oxide_competent: NDArray[np.bool_],
    ) -> SlowMemoryState:
        """Create a baseline state for a fixed set of independently tracked routes."""

        competent = np.asarray(nitric_oxide_competent, dtype=np.bool_)
        if competent.ndim != 1:
            raise ValueError("nitric_oxide_competent must be one-dimensional")
        baseline = np.zeros(competent.size, dtype=np.float64)
        return cls(
            nitric_oxide_competent=competent.copy(),
            dopamine_immediate=baseline.copy(),
            nitric_oxide_immediate=baseline.copy(),
            dopamine_effect=baseline.copy(),
            nitric_oxide_effect=baseline.copy(),
        )

    def step(
        self,
        *,
        paired: NDArray[np.bool_],
        dan_unpaired: NDArray[np.bool_],
        dt_seconds: float,
        parameters: SlowMemoryParameters,
    ) -> None:
        """Advance measured induction and effect dynamics without changing anatomy."""

        parameters.validate()
        if not math.isfinite(dt_seconds) or dt_seconds < 0:
            raise ValueError("dt_seconds must be finite and non-negative")
        paired_mask = np.asarray(paired, dtype=np.bool_)
        unpaired_mask = np.asarray(dan_unpaired, dtype=np.bool_)
        expected_shape = self.nitric_oxide_competent.shape
        if paired_mask.ndim != 1 or paired_mask.shape != expected_shape:
            raise ValueError("paired must match the one-dimensional state shape")
        if unpaired_mask.ndim != 1 or unpaired_mask.shape != expected_shape:
            raise ValueError("dan_unpaired must match the one-dimensional state shape")
        if np.any(paired_mask & unpaired_mask):
            raise ValueError("paired and dan_unpaired activation must not overlap")
        self._validate_state()

        dt_minutes = dt_seconds / 60.0
        dopamine_induction = 1.0 - math.exp(
            -parameters.dopamine_induction_per_minute * dt_minutes
        )
        nitric_oxide_induction = 1.0 - math.exp(
            -parameters.nitric_oxide_induction_per_minute * dt_minutes
        )
        self.dopamine_immediate[paired_mask] += (
            1.0 - self.dopamine_immediate[paired_mask]
        ) * dopamine_induction
        no_paired = paired_mask & self.nitric_oxide_competent
        self.nitric_oxide_immediate[no_paired] += (
            1.0 - self.nitric_oxide_immediate[no_paired]
        ) * nitric_oxide_induction

        self.dopamine_immediate[unpaired_mask] *= math.exp(
            -parameters.dopamine_unpaired_decay_per_minute * dt_minutes
        )
        no_unpaired = unpaired_mask & self.nitric_oxide_competent
        self.nitric_oxide_immediate[no_unpaired] *= math.exp(
            -parameters.nitric_oxide_unpaired_decay_per_minute * dt_minutes
        )
        self.nitric_oxide_immediate[~self.nitric_oxide_competent] = 0.0

        dopamine_tracking = 1.0 - math.exp(
            -dt_seconds / parameters.dopamine_effect_tau_seconds
        )
        nitric_oxide_tracking = 1.0 - math.exp(
            -dt_seconds / parameters.nitric_oxide_effect_tau_seconds
        )
        self.dopamine_effect += (
            self.dopamine_immediate - self.dopamine_effect
        ) * dopamine_tracking
        self.nitric_oxide_effect += (
            self.nitric_oxide_immediate - self.nitric_oxide_effect
        ) * nitric_oxide_tracking
        self.nitric_oxide_effect[~self.nitric_oxide_competent] = 0.0

    @property
    def weight_multipliers(self) -> NDArray[np.float64]:
        """Return the Aso 2019 multiplicative combination ``(1-D)(1+N)``."""

        self._validate_state()
        return (1.0 - self.dopamine_effect) * (1.0 + self.nitric_oxide_effect)

    def _validate_state(self) -> None:
        arrays = (
            self.dopamine_immediate,
            self.nitric_oxide_immediate,
            self.dopamine_effect,
            self.nitric_oxide_effect,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("slow-memory state arrays must be one-dimensional")
        if any(array.shape != self.nitric_oxide_competent.shape for array in arrays):
            raise ValueError("slow-memory state arrays must have equal shape")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("slow-memory state arrays must be finite")
        if any(np.any((array < 0) | (array > 1)) for array in arrays):
            raise ValueError("slow-memory state values must lie within [0, 1]")
