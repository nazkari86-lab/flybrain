"""Measured ORN-type channels for structured, nonsemantic olfactory input."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from flybrain.graph import EventConnectome


class OlfactoryReceptorBank(BaseModel, frozen=True):
    """One exact MaleCNS ORN cell-type channel."""

    model_config = ConfigDict(extra="forbid")

    cell_type: str = Field(pattern=r"^ORN_[A-Za-z0-9]+$")
    neuron_ids: tuple[int, ...]


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

    def channel_ids(self, cell_types: tuple[str, ...]) -> tuple[int, ...]:
        """Return the exact union for a predeclared receptor-channel pattern."""

        if not cell_types or len(set(cell_types)) != len(cell_types):
            raise ValueError("receptor channel types must be nonempty and unique")
        by_type = {bank.cell_type: bank.neuron_ids for bank in self.banks}
        absent = sorted(set(cell_types) - set(by_type))
        if absent:
            raise ValueError(f"unknown ORN channel types: {absent}")
        return tuple(sorted(neuron_id for item in cell_types for neuron_id in by_type[item]))
