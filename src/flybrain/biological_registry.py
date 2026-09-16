"""Evidence-bound biological population declarations and exact snapshot resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from flybrain.graph import EventConnectome
from flybrain.provenance import snapshot_content_sha256

EvidenceKind = Literal[
    "dataset_measurement",
    "experimental_result",
    "model_assumption",
    "simulation_observation",
]
ALLOWED_SELECTOR_COLUMNS = frozenset(
    {
        "type",
        "superclass",
        "class",
        "subclass",
        "somaSide",
        "rootSide",
        "entryNerve",
        "exitNerve",
        "receptorType",
    }
)
_LATERALITY_COLUMNS = frozenset({"somaSide", "rootSide"})
_LATERALITY_VALUES = frozenset({"L", "R"})


class EvidenceRecord(BaseModel, frozen=True):
    """One typed source supporting exactly one registry claim."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    source_url: AnyHttpUrl
    claim: str = Field(min_length=1)
    pmid: str | None = None
    doi: str | None = None
    confidence: Literal["measured", "high", "moderate", "assumption"]
    source_artifact_name: str | None = None
    source_artifact_url: AnyHttpUrl | None = None
    source_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("pmid", mode="before")
    @classmethod
    def validate_pmid(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not str or re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise ValueError("PMID must be a nonblank positive decimal identifier")
        return value

    @field_validator("doi", mode="before")
    @classmethod
    def validate_doi(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not str or re.fullmatch(r"10\.[0-9]{4,9}/\S+", value) is None:
            raise ValueError("DOI must be a nonblank DOI identifier")
        return value

    @model_validator(mode="after")
    def validate_evidence_semantics(self) -> Self:
        """Prevent claims from crossing the measured/assumed evidence boundary."""

        if self.kind == "experimental_result" and self.pmid is None and self.doi is None:
            raise ValueError("experimental-result evidence requires a PMID or DOI")
        if self.kind == "model_assumption" and self.confidence != "assumption":
            raise ValueError('model-assumption evidence requires confidence="assumption"')
        if self.kind != "model_assumption" and self.confidence == "assumption":
            raise ValueError("non-assumption evidence cannot use assumption confidence")
        artifact_fields = (
            self.source_artifact_name,
            self.source_artifact_url,
            self.source_artifact_sha256,
        )
        if any(value is not None for value in artifact_fields) and not all(
            value is not None for value in artifact_fields
        ):
            raise ValueError(
                "source artifact name, URL, and SHA-256 must be supplied together"
            )
        return self


class AnnotationSelector(BaseModel, frozen=True):
    """Allowlisted exact predicates and optional retained-snapshot assertions."""

    model_config = ConfigDict(extra="forbid")

    equals: Mapping[str, str]
    in_values: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict,
        validate_default=True,
    )
    expected_ids: tuple[int, ...] = ()
    expected_count: int | None = Field(default=None, gt=0)

    @field_validator("expected_ids", mode="before")
    @classmethod
    def validate_expected_id_types(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not int for item in value
        ):
            raise ValueError("expected IDs must contain only integers")
        return value

    @field_validator("expected_count", mode="before")
    @classmethod
    def validate_expected_count_type(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("expected count must be an integer")
        return value

    @field_validator("equals", mode="after")
    @classmethod
    def freeze_equals(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_validator("in_values", mode="after")
    @classmethod
    def freeze_in_values(
        cls,
        value: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(dict(value))

    @field_serializer("equals")
    def serialize_equals(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_serializer("in_values")
    def serialize_in_values(
        self,
        value: Mapping[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        return dict(value)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        """Reject selectors that are ambiguous, unstable, or biologically invalid."""

        columns = set(self.equals) | set(self.in_values)
        if not columns:
            raise ValueError("annotation selector requires at least one predicate")
        unknown = sorted(columns - ALLOWED_SELECTOR_COLUMNS)
        if unknown:
            raise ValueError(f"selector column is not allowlisted: {unknown}")
        conflicts = sorted(set(self.equals) & set(self.in_values))
        if conflicts:
            raise ValueError(f"conflicting equals/in_values selector columns: {conflicts}")
        for column, values in self.in_values.items():
            if not values:
                raise ValueError(f"selector predicate has no values: {column}")
            if len(values) != len(set(values)):
                raise ValueError(f"selector predicate contains duplicate values: {column}")
        for column in columns & _LATERALITY_COLUMNS:
            values = (
                (self.equals[column],)
                if column in self.equals
                else self.in_values[column]
            )
            unexpected = sorted(set(values) - _LATERALITY_VALUES)
            if unexpected:
                raise ValueError(f"unexpected laterality for {column}: {unexpected}")
        if any(type(value) is not int or value <= 0 for value in self.expected_ids):
            raise ValueError("expected IDs must be positive integers")
        if tuple(sorted(set(self.expected_ids))) != self.expected_ids:
            raise ValueError("expected IDs must be unique and sorted")
        if (
            self.expected_ids
            and self.expected_count is not None
            and len(self.expected_ids) != self.expected_count
        ):
            raise ValueError("expected IDs and expected count disagree")
        return self


class PopulationDeclaration(BaseModel, frozen=True):
    """A named biological role selected from canonical annotations."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: Literal["sensory", "steering", "retreat", "future_interface"]
    selector: AnnotationSelector
    evidence_ids: tuple[str, ...]


class BiologicalInterfaceRegistry(BaseModel, frozen=True):
    """Versioned declarations and their complete typed evidence catalog."""

    model_config = ConfigDict(extra="forbid")

    registry_version: str
    dataset_id: str
    populations: tuple[PopulationDeclaration, ...]
    evidence: tuple[EvidenceRecord, ...]

    @model_validator(mode="after")
    def validate_registry_references(self) -> Self:
        """Require unique populations, unique evidence, and closed references."""

        names = tuple(population.name for population in self.populations)
        if len(names) != len(set(names)):
            raise ValueError("duplicate population name")
        evidence_ids = tuple(record.evidence_id for record in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence ID")
        available = set(evidence_ids)
        for population in self.populations:
            if not population.evidence_ids:
                raise ValueError(f"population has no evidence references: {population.name}")
            if len(population.evidence_ids) != len(set(population.evidence_ids)):
                raise ValueError(f"duplicate evidence reference: {population.name}")
            absent = sorted(set(population.evidence_ids) - available)
            if absent:
                raise ValueError(
                    f"absent evidence reference for population {population.name}: {absent}"
                )
        return self


class ResolvedPopulation(BaseModel, frozen=True):
    """One declaration resolved to exact, stable canonical neuron IDs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    selector: AnnotationSelector
    neuron_ids: tuple[int, ...]
    id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[EvidenceRecord, ...]


class ResolvedRegistry(BaseModel, frozen=True):
    """Registry result bound to current registry, snapshot, and annotation bytes."""

    model_config = ConfigDict(extra="forbid")

    registry_version: str
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    populations: tuple[ResolvedPopulation, ...]

    def population(self, name: str) -> ResolvedPopulation:
        """Return a uniquely named resolved population or fail closed."""

        matches = tuple(item for item in self.populations if item.name == name)
        if len(matches) != 1:
            raise ValueError(f"resolved population not found exactly once: {name}")
        return matches[0]

    def validate_graph(self, graph: EventConnectome) -> None:
        """Require every resolved interface ID to exist in the executed graph."""

        available = {int(value) for value in graph.neuron_ids}
        required = {
            neuron_id
            for population in self.populations
            for neuron_id in population.neuron_ids
        }
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"resolved population IDs absent from graph: {missing}")


class _SnapshotMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dataset_id: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_biological_registry(path: Path) -> BiologicalInterfaceRegistry:
    """Load and strictly validate a declarative biological interface registry."""

    return BiologicalInterfaceRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def _canonical_registry_bytes(registry: BiologicalInterfaceRegistry) -> bytes:
    payload = registry.model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_body_ids(table: pa.Table) -> tuple[int, ...]:
    raw_ids = table.column("bodyId").to_pylist()
    if any(type(value) is not int or value <= 0 for value in raw_ids):
        raise ValueError("source bodyId values must be positive integers")
    body_ids = tuple(int(value) for value in raw_ids)
    if len(body_ids) != len(set(body_ids)):
        raise ValueError("duplicate source bodyId")
    return body_ids


def _resolve_population_ids(
    table: pa.Table,
    declaration: PopulationDeclaration,
) -> tuple[int, ...]:
    mask: pa.Array | pa.ChunkedArray | None = None
    for column_name, exact_value in sorted(declaration.selector.equals.items()):
        predicate = pc.equal(table.column(column_name), pa.scalar(exact_value))
        mask = predicate if mask is None else pc.and_(mask, predicate)
    for column_name, accepted_values in sorted(declaration.selector.in_values.items()):
        column = table.column(column_name)
        predicate = pc.is_in(column, value_set=pa.array(accepted_values, type=column.type))
        mask = predicate if mask is None else pc.and_(mask, predicate)
    if mask is None:
        raise ValueError(f"population selector is empty: {declaration.name}")
    selected = table.filter(pc.fill_null(mask, False)).column("bodyId").to_pylist()
    neuron_ids = tuple(sorted({int(value) for value in selected}))
    if not neuron_ids:
        raise ValueError(f"resolved population is empty: {declaration.name}")
    if declaration.selector.expected_ids and neuron_ids != declaration.selector.expected_ids:
        raise ValueError(
            f"resolved population expected IDs differ for {declaration.name}: {neuron_ids}"
        )
    if (
        declaration.selector.expected_count is not None
        and len(neuron_ids) != declaration.selector.expected_count
    ):
        raise ValueError(
            "resolved population expected count differs for "
            f"{declaration.name}: {len(neuron_ids)}"
        )
    return neuron_ids


def resolve_biological_registry(
    registry: BiologicalInterfaceRegistry,
    snapshot: Path,
) -> ResolvedRegistry:
    """Resolve every declaration exactly and bind it to current snapshot identities."""

    metadata = _SnapshotMetadata.model_validate_json(
        (snapshot / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.dataset_id != registry.dataset_id:
        raise ValueError(
            "registry dataset_id does not match snapshot metadata: "
            f"{registry.dataset_id} != {metadata.dataset_id}"
        )

    annotations_path = snapshot / "source-annotations.parquet"
    schema = pq.read_schema(annotations_path)
    required_columns = {"bodyId"}
    for declaration in registry.populations:
        required_columns.update(declaration.selector.equals)
        required_columns.update(declaration.selector.in_values)
    missing_columns = sorted(required_columns - set(schema.names))
    if missing_columns:
        raise ValueError(f"missing annotation columns: {', '.join(missing_columns)}")

    table = pq.read_table(annotations_path, columns=sorted(required_columns))
    _validated_body_ids(table)
    evidence_by_id = {record.evidence_id: record for record in registry.evidence}
    resolved_populations = []
    for declaration in registry.populations:
        neuron_ids = _resolve_population_ids(table, declaration)
        id_bytes = b"\n".join(str(value).encode("ascii") for value in neuron_ids)
        resolved_populations.append(
            ResolvedPopulation(
                name=declaration.name,
                role=declaration.role,
                selector=declaration.selector,
                neuron_ids=neuron_ids,
                id_sha256=hashlib.sha256(id_bytes).hexdigest(),
                evidence=tuple(evidence_by_id[item] for item in declaration.evidence_ids),
            )
        )

    canonical_registry = _canonical_registry_bytes(registry)
    return ResolvedRegistry(
        registry_version=registry.registry_version,
        registry_sha256=hashlib.sha256(canonical_registry).hexdigest(),
        dataset_id=registry.dataset_id,
        source_manifest_sha256=metadata.manifest_sha256,
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        annotation_sha256=_file_sha256(annotations_path),
        populations=tuple(resolved_populations),
    )
