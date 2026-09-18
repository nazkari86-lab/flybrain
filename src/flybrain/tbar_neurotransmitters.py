"""Audit presynaptic transmitter probabilities without changing connectome weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc


@dataclass(frozen=True)
class TbarBodyTransmitterSummary:
    """T-bar probability mean for one exact presynaptic MaleCNS body ID."""

    body_id: int
    tbar_count: int
    mean_probabilities: dict[str, float]

    @property
    def dominant_transmitter(self) -> str:
        """Return the highest-probability transmitter label, deterministically."""

        return min(
            self.mean_probabilities,
            key=lambda name: (-self.mean_probabilities[name], name),
        )


@dataclass(frozen=True)
class TbarNeurotransmitterAudit:
    """A provenance-neutral measurement summary of one T-bar prediction export."""

    source_tbar_rows: int
    matched_tbar_rows: int
    missing_body_ids: tuple[int, ...]
    bodies: tuple[TbarBodyTransmitterSummary, ...]


@dataclass(frozen=True)
class BodyTransmitterSummary:
    """One official body-level prediction and its release consensus label."""

    body_id: int
    consensus_transmitter: str
    predicted_transmitter: str
    prediction_confidence: float


@dataclass(frozen=True)
class BodyNeurotransmitterAudit:
    """Exact-ID audit of the official aggregate MaleCNS transmitter product."""

    source_body_rows: int
    missing_body_ids: tuple[int, ...]
    consensus_prediction_disagreements: int
    bodies: tuple[BodyTransmitterSummary, ...]


def audit_tbar_neurotransmitters(
    source: Path,
    *,
    body_ids: tuple[int, ...],
) -> TbarNeurotransmitterAudit:
    """Stream source T-bars and aggregate probabilities only for exact body IDs.

    The result is a measurement audit. It does not alter consensus transmitter
    labels, signs, contact counts, or any simulator conductance.
    """

    requested = tuple(sorted(set(body_ids)))
    if not requested:
        raise ValueError("body_ids must not be empty")

    with pa.memory_map(str(source), "r") as mapped:
        reader = ipc.open_file(mapped)
        names = set(reader.schema.names)
        if "body" not in names:
            raise ValueError("T-bar source requires body")
        probability_columns = tuple(
            name.removeprefix("nt_").removesuffix("_prob")
            for name in reader.schema.names
            if name.startswith("nt_") and name.endswith("_prob")
        )
        if not probability_columns:
            raise ValueError("T-bar source requires at least one nt_*_prob column")

        source_rows = 0
        matched_rows = 0
        counts = dict.fromkeys(requested, 0)
        totals = {
            body_id: dict.fromkeys(probability_columns, 0.0) for body_id in requested
        }
        requested_values = pa.array(requested, type=pa.uint64())

        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            source_rows += batch.num_rows
            selected = batch.filter(
                pc.is_in(batch.column("body"), value_set=requested_values)
            )
            matched_rows += selected.num_rows
            if not selected.num_rows:
                continue
            bodies = selected.column("body").to_pylist()
            values = {
                transmitter: selected.column(f"nt_{transmitter}_prob").to_pylist()
                for transmitter in probability_columns
            }
            for row_index, body in enumerate(bodies):
                body_id = int(body)
                counts[body_id] += 1
                for transmitter in probability_columns:
                    totals[body_id][transmitter] += float(values[transmitter][row_index])

    summaries = tuple(
        TbarBodyTransmitterSummary(
            body_id=body_id,
            tbar_count=counts[body_id],
            mean_probabilities={
                transmitter: total / counts[body_id]
                for transmitter, total in totals[body_id].items()
            },
        )
        for body_id in requested
        if counts[body_id]
    )
    return TbarNeurotransmitterAudit(
        source_tbar_rows=source_rows,
        matched_tbar_rows=matched_rows,
        missing_body_ids=tuple(body_id for body_id in requested if not counts[body_id]),
        bodies=summaries,
    )


def audit_body_neurotransmitters(
    source: Path,
    *,
    body_ids: tuple[int, ...],
) -> BodyNeurotransmitterAudit:
    """Read Janelia's official aggregation without substituting raw T-bar means.

    ``consensus_nt`` remains the release's authoritative neuron-level label;
    ``predicted_nt`` is retained solely to expose disagreement and uncertainty.
    """

    requested = tuple(sorted(set(body_ids)))
    if not requested:
        raise ValueError("body_ids must not be empty")
    table = feather.read_table(str(source), memory_map=True)
    required = {
        "body",
        "predicted_nt",
        "predicted_nt_confidence",
        "consensus_nt",
    }
    missing_columns = sorted(required - set(table.column_names))
    if missing_columns:
        raise ValueError(f"body transmitter source missing columns: {missing_columns}")
    selected = table.filter(
        pc.is_in(table.column("body"), value_set=pa.array(requested, type=pa.uint64()))
    )
    records = {
        int(body): BodyTransmitterSummary(
            body_id=int(body),
            consensus_transmitter=str(consensus),
            predicted_transmitter=str(predicted),
            prediction_confidence=float(confidence),
        )
        for body, predicted, confidence, consensus in zip(
            selected.column("body").to_pylist(),
            selected.column("predicted_nt").to_pylist(),
            selected.column("predicted_nt_confidence").to_pylist(),
            selected.column("consensus_nt").to_pylist(),
            strict=True,
        )
    }
    if len(records) != selected.num_rows:
        raise ValueError("body transmitter source has duplicate body IDs")
    summaries = tuple(records[body_id] for body_id in requested if body_id in records)
    return BodyNeurotransmitterAudit(
        source_body_rows=table.num_rows,
        missing_body_ids=tuple(body_id for body_id in requested if body_id not in records),
        consensus_prediction_disagreements=sum(
            item.consensus_transmitter != item.predicted_transmitter for item in summaries
        ),
        bodies=summaries,
    )
