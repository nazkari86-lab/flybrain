"""Measured ORN-type channels for structured, nonsemantic olfactory input."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.graph import EventConnectome


class TaskOdorAssignment(BaseModel, frozen=True):
    """Literature-backed adult channels used to instantiate task odor sources."""

    model_config = ConfigDict(extra="forbid")

    food_cell_types: tuple[str, ...]
    threat_cell_types: tuple[str, ...]
    food_evidence_doi: str
    threat_evidence_doi: str
    evidence_scope: str = "adult_innate_valence"

    @model_validator(mode="after")
    def validate_assignment(self) -> TaskOdorAssignment:
        if not self.food_cell_types or not self.threat_cell_types:
            raise ValueError("task odor assignments must be nonempty")
        if set(self.food_cell_types) & set(self.threat_cell_types):
            raise ValueError("food and threat odor channels must be disjoint")
        return self


def task_odor_assignment() -> TaskOdorAssignment:
    """Return adult glomerular channels with causal innate-valence evidence.

    DM1 and VA2 are each necessary/sufficient contributors to vinegar attraction
    (Semmelhack & Wang, 2009). Or56a sensory neurons project to DA2 and form a
    dedicated geosmin avoidance channel (Stensmyr et al., 2012).
    """

    return TaskOdorAssignment(
        food_cell_types=("ORN_DM1", "ORN_VA2"),
        threat_cell_types=("ORN_DA2",),
        food_evidence_doi="10.1038/nature07983",
        threat_evidence_doi="10.1016/j.cell.2012.09.046",
    )


class OlfactoryReceptorBank(BaseModel, frozen=True):
    """One exact MaleCNS ORN cell-type channel."""

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(pattern=r"^ORN_[A-Za-z0-9]+$")
    neuron_ids: tuple[int, ...]
    left_ids: tuple[int, ...] = ()
    right_ids: tuple[int, ...] = ()
    unknown_side_ids: tuple[int, ...] = ()


class OlfactoryReceptorMap(BaseModel, frozen=True):
    """A topology-preserving mapping from ORN types to retained neuron IDs."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: str = "dataset_measurement"
    banks: tuple[OlfactoryReceptorBank, ...]
    excluded_untyped_neuron_ids: tuple[int, ...] = ()

    @classmethod
    def from_graph(
        cls,
        graph: EventConnectome,
        *,
        olfactory_neuron_ids: tuple[int, ...],
    ) -> OlfactoryReceptorMap:
        """Resolve only declared ORN types in one exact sensory population."""

        if not olfactory_neuron_ids or len(set(olfactory_neuron_ids)) != len(
            olfactory_neuron_ids
        ):
            raise ValueError("olfactory neuron IDs must be nonempty and unique")
        index = {int(neuron_id): position for position, neuron_id in enumerate(graph.neuron_ids)}
        grouped: dict[str, list[int]] = {}
        excluded: list[int] = []
        for neuron_id in olfactory_neuron_ids:
            try:
                cell_type = graph.cell_types[index[neuron_id]]
            except KeyError as error:
                raise ValueError(f"olfactory neuron absent from graph: {neuron_id}") from error
            if not cell_type.startswith("ORN_"):
                excluded.append(neuron_id)
                continue
            grouped.setdefault(cell_type, []).append(neuron_id)
        return cls(
            banks=tuple(
                OlfactoryReceptorBank(
                    cell_type=cell_type,
                    neuron_ids=tuple(sorted(neuron_ids)),
                )
                for cell_type, neuron_ids in sorted(grouped.items())
            ),
            excluded_untyped_neuron_ids=tuple(sorted(excluded)),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Path,
        *,
        graph: EventConnectome,
        olfactory_neuron_ids: tuple[int, ...],
    ) -> OlfactoryReceptorMap:
        """Resolve receptor type and measured root-side from retained annotations."""

        base = cls.from_graph(graph, olfactory_neuron_ids=olfactory_neuron_ids)
        table = pq.read_table(
            snapshot / "source-annotations.parquet", columns=["bodyId", "rootSide"]
        )
        side_by_id = {
            int(body_id): str(side) if side is not None else "unknown"
            for body_id, side in zip(
                table.column("bodyId").to_pylist(),
                table.column("rootSide").to_pylist(),
                strict=True,
            )
        }
        banks = []
        for bank in base.banks:
            left = tuple(sorted(item for item in bank.neuron_ids if side_by_id.get(item) == "L"))
            right = tuple(sorted(item for item in bank.neuron_ids if side_by_id.get(item) == "R"))
            unknown = tuple(item for item in bank.neuron_ids if item not in set(left) | set(right))
            banks.append(
                bank.model_copy(
                    update={
                        "left_ids": left,
                        "right_ids": right,
                        "unknown_side_ids": unknown,
                    }
                )
            )
        return cls(banks=tuple(banks), excluded_untyped_neuron_ids=base.excluded_untyped_neuron_ids)

    def channel_ids(self, cell_types: tuple[str, ...]) -> tuple[int, ...]:
        """Return the exact union for a predeclared receptor-channel pattern."""

        if not cell_types or len(set(cell_types)) != len(cell_types):
            raise ValueError("receptor channel types must be nonempty and unique")
        by_type = {bank.cell_type: bank.neuron_ids for bank in self.banks}
        absent = sorted(set(cell_types) - set(by_type))
        if absent:
            raise ValueError(f"unknown ORN channel types: {absent}")
        return tuple(sorted(neuron_id for item in cell_types for neuron_id in by_type[item]))

    def side_channel_ids(
        self, cell_types: tuple[str, ...], side: str
    ) -> tuple[int, ...]:
        """Return only annotations measured on one root side (L or R)."""

        if side not in {"L", "R"}:
            raise ValueError("side must be L or R")
        selected: list[int] = []
        for bank in self.banks:
            if bank.cell_type in cell_types:
                selected.extend(bank.left_ids if side == "L" else bank.right_ids)
        if not selected:
            raise ValueError(f"no measured ORN IDs for side {side}")
        return tuple(sorted(selected))
