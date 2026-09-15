"""Streaming adapter for official MaleCNS flat-connectome exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc

from flybrain.acquire import acquire_artifact
from flybrain.manifest import load_manifest
from flybrain.schema import NEURON_SCHEMA


@dataclass(frozen=True)
class MaleCNSSources:
    """Verified local paths for the four essential MaleCNS products."""

    annotations: Path
    neurotransmitters: Path
    stats: Path
    weights: Path
    manifest_sha256: str
    dataset_id: str

    @classmethod
    def from_manifest(cls, manifest_path: Path, cache_root: Path) -> MaleCNSSources:
        manifest = load_manifest(manifest_path)
        resolved: dict[str, Path] = {}
        for artifact in manifest.artifacts:
            filename = Path(urlparse(str(artifact.url)).path).name
            if filename.startswith("body-annotations-"):
                key = "annotations"
            elif filename.startswith("body-neurotransmitters-"):
                key = "neurotransmitters"
            elif filename.startswith("body-stats-"):
                key = "stats"
            elif filename.startswith("connectome-weights-"):
                key = "weights"
            else:
                continue
            if key in resolved:
                raise ValueError(f"duplicate MaleCNS artifact role: {key}")
            resolved[key] = acquire_artifact(artifact, cache_root)

        required = {"annotations", "neurotransmitters", "stats", "weights"}
        missing = sorted(required - resolved.keys())
        if missing:
            raise ValueError(f"missing MaleCNS artifacts: {', '.join(missing)}")

        import hashlib

        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return cls(
            annotations=resolved["annotations"],
            neurotransmitters=resolved["neurotransmitters"],
            stats=resolved["stats"],
            weights=resolved["weights"],
            manifest_sha256=manifest_sha256,
            dataset_id=manifest.dataset_id,
        )


@dataclass(frozen=True)
class MaleCNSSelectionMetrics:
    """Measured population counts from the source annotation policy."""

    source_annotation_rows: int
    selected_neurons: int
    untyped_neurons: int
    missing_transmitters: int


def transmitter_sign(name: str) -> tuple[int, str]:
    """Return a conservative fast-sign assumption and its provenance."""

    normalized = name.strip().lower()
    if normalized == "acetylcholine":
        return 1, "fast-transmitter-assumption"
    if normalized in {"gaba", "histamine"}:
        return -1, "fast-transmitter-assumption"
    if normalized == "glutamate":
        return 0, "receptor-context-required"
    if normalized in {"dopamine", "octopamine", "serotonin"}:
        return 0, "neuromodulator-not-fast-sign"
    return 0, "unresolved-transmitter"


def _text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _role(superclass: str) -> str:
    if superclass.endswith("_sensory"):
        return "sensory"
    if superclass == "vnc_motor":
        return "motor"
    if superclass == "descending_neuron":
        return "descending"
    if superclass == "ascending_neuron":
        return "ascending"
    return "interneuron"


def select_neurons(
    annotations: pa.Table,
    neurotransmitters: pa.Table,
    *,
    dataset_id: str = "male-cns-v1.0",
) -> tuple[pa.Table, MaleCNSSelectionMetrics]:
    """Apply the published valid-superclass policy and join biological metadata."""

    required_annotations = {
        "bodyId",
        "type",
        "superclass",
        "somaSide",
        "rootSide",
        "status",
        "statusLabel",
    }
    missing_annotations = sorted(required_annotations - set(annotations.column_names))
    if missing_annotations:
        raise ValueError(f"missing annotation columns: {', '.join(missing_annotations)}")
    if not {"body", "consensus_nt"}.issubset(neurotransmitters.column_names):
        raise ValueError("neurotransmitter table requires body and consensus_nt")

    superclasses = annotations.column("superclass").to_pylist()
    valid_mask = pa.array(
        [value is not None and "tbc" not in str(value).lower() for value in superclasses]
    )
    selected = annotations.filter(valid_mask)
    selected_ids = selected.column("bodyId").to_pylist()
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate selected annotation body ID")

    selected_nt = neurotransmitters.filter(
        pc.is_in(neurotransmitters.column("body"), value_set=pa.array(selected_ids))
    )
    nt_bodies = selected_nt.column("body").to_pylist()
    if len(nt_bodies) != len(set(nt_bodies)):
        raise ValueError("duplicate neurotransmitter body ID")
    nt_by_body = dict(
        zip(nt_bodies, selected_nt.column("consensus_nt").to_pylist(), strict=True)
    )

    records = selected.to_pylist()
    untyped = 0
    missing_transmitters = 0
    columns: dict[str, list[object]] = {name: [] for name in NEURON_SCHEMA.names}
    for record in records:
        body_id = int(record["bodyId"])
        cell_type = _text(record["type"], "untyped")
        if cell_type == "untyped":
            untyped += 1
        superclass = _text(record["superclass"], "unknown")
        soma_side = _text(record["somaSide"], "")
        side = soma_side or _text(record["rootSide"], "unknown")
        raw_transmitter = nt_by_body.get(body_id)
        transmitter = _text(raw_transmitter, "unclear")
        if raw_transmitter is None:
            missing_transmitters += 1
        columns["neuron_id"].append(body_id)
        columns["source_dataset"].append(dataset_id)
        columns["cell_type"].append(cell_type)
        columns["superclass"].append(superclass)
        columns["side"].append(side)
        columns["transmitter"].append(transmitter)
        columns["transmitter_provenance"].append(
            "male-cns-consensus" if raw_transmitter is not None else "missing-source-annotation"
        )
        columns["role"].append(_role(superclass))
        columns["annotation_status"].append(_text(record["status"], "unknown"))
        columns["status_label"].append(_text(record["statusLabel"], "unknown"))
        columns["annotation_confidence"].append(0.5)

    neurons = pa.Table.from_pydict(columns, schema=NEURON_SCHEMA).sort_by("neuron_id")
    metrics = MaleCNSSelectionMetrics(
        source_annotation_rows=annotations.num_rows,
        selected_neurons=neurons.num_rows,
        untyped_neurons=untyped,
        missing_transmitters=missing_transmitters,
    )
    return neurons, metrics
