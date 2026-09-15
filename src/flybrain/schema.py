"""Canonical columnar schemas for versioned connectome snapshots."""

import pyarrow as pa
from pydantic import BaseModel, Field

NEURON_SCHEMA = pa.schema(
    [
        ("neuron_id", pa.uint64()),
        ("source_dataset", pa.string()),
        ("cell_type", pa.string()),
        ("superclass", pa.string()),
        ("side", pa.string()),
        ("transmitter", pa.string()),
        ("transmitter_provenance", pa.string()),
        ("role", pa.string()),
        ("annotation_status", pa.string()),
        ("status_label", pa.string()),
        ("annotation_confidence", pa.float32()),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        ("pre_id", pa.uint64()),
        ("post_id", pa.uint64()),
        ("synapse_count", pa.uint32()),
        ("sign", pa.int8()),
        ("sign_provenance", pa.string()),
        ("confidence", pa.float32()),
    ]
)


class SnapshotMetadata(BaseModel):
    """Exact provenance of one canonical snapshot transformation."""

    dataset_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    importer: str = Field(min_length=1)
