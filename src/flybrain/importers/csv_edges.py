"""Importer for tabular neuron and directed-edge exports."""

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from flybrain.schema import EDGE_SCHEMA, NEURON_SCHEMA, SnapshotMetadata


class SnapshotIntegrityError(ValueError):
    """Source rows violate canonical graph integrity."""


def _read_canonical_csv(path: Path, schema: pa.Schema) -> pa.Table:
    table = pacsv.read_csv(path)
    try:
        return table.select(schema.names).cast(schema)
    except (KeyError, pa.ArrowInvalid) as error:
        raise SnapshotIntegrityError(f"invalid canonical columns in {path}: {error}") from error


def import_csv_snapshot(
    neurons_path: Path,
    edges_path: Path,
    output: Path,
    metadata: SnapshotMetadata,
) -> Path:
    """Validate tabular source files and write one canonical snapshot."""

    neurons = _read_canonical_csv(neurons_path, NEURON_SCHEMA)
    edges = _read_canonical_csv(edges_path, EDGE_SCHEMA)

    neuron_ids = neurons.column("neuron_id").to_pylist()
    if len(neuron_ids) != len(set(neuron_ids)):
        raise SnapshotIntegrityError("duplicate neuron ID")

    known = set(neuron_ids)
    for column_name in ("pre_id", "post_id"):
        for neuron_id in edges.column(column_name).to_pylist():
            if neuron_id not in known:
                raise SnapshotIntegrityError(f"unknown neuron {neuron_id}")

    signs = set(edges.column("sign").to_pylist())
    if not signs.issubset({-1, 0, 1}):
        raise SnapshotIntegrityError("edge sign must be -1, 0, or 1")

    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(neurons, output / "neurons.parquet")
    pq.write_table(edges, output / "edges.parquet")
    (output / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    return output
