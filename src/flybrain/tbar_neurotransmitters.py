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


@dataclass(frozen=True)
class ContactTransmitterSummary:
    """Raw T-bar probabilities joined to one exact anatomical edge's contacts."""

    pre_id: int
    post_id: int
    contact_count: int
    matched_tbar_count: int
    mean_probabilities: dict[str, float]


@dataclass(frozen=True)
class ContactTransmitterAudit:
    """Contact-level measurement; it does not change any neuron-level consensus."""

    partner_contact_rows: int
    matched_contact_rows: int
    unmatched_contact_rows: int
    edges: tuple[ContactTransmitterSummary, ...]


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


def audit_contact_transmitters(
    tbar_source: Path,
    partner_source: Path,
    *,
    edge_pairs: tuple[tuple[int, int], ...],
) -> ContactTransmitterAudit:
    """Join raw T-bar predictions to declared contacts by exact pre coordinates.

    Contact coordinate matching is a measurement of raw prediction uncertainty.
    It must not overwrite the official body-level ``consensus_nt`` label.
    """

    declared = tuple(sorted(set(edge_pairs)))
    if not declared or any(
        type(pre_id) is not int or type(post_id) is not int or pre_id <= 0 or post_id <= 0
        for pre_id, post_id in declared
    ):
        raise ValueError("edge_pairs must contain positive integer endpoints")
    if len(declared) != len(edge_pairs):
        raise ValueError("edge_pairs must be unique")
    declared_set = frozenset(declared)
    pre_values = pa.array(sorted({pre_id for pre_id, _ in declared}), type=pa.uint64())
    contact_coordinates: dict[tuple[int, int], list[tuple[int, int, int]]] = {
        pair: [] for pair in declared
    }

    with pa.memory_map(str(partner_source), "r") as mapped:
        reader = ipc.open_file(mapped)
        required = {"x_pre", "y_pre", "z_pre", "body_pre", "body_post"}
        missing = sorted(required - set(reader.schema.names))
        if missing:
            raise ValueError(f"partner source missing columns: {missing}")
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            selected = batch.filter(
                pc.is_in(batch.column("body_pre"), value_set=pre_values)
            )
            for pre_id, post_id, x, y, z in zip(
                selected.column("body_pre").to_pylist(),
                selected.column("body_post").to_pylist(),
                selected.column("x_pre").to_pylist(),
                selected.column("y_pre").to_pylist(),
                selected.column("z_pre").to_pylist(),
                strict=True,
            ):
                pair = (int(pre_id), int(post_id))
                if pair in declared_set:
                    contact_coordinates[pair].append((int(x), int(y), int(z)))

    pairs_by_coordinate: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for pair, coordinates in contact_coordinates.items():
        for coordinate in coordinates:
            pairs_by_coordinate.setdefault(coordinate, []).append(pair)
    probability_totals: dict[tuple[int, int], dict[str, float]] = {}
    matched_counts = dict.fromkeys(declared, 0)

    with pa.memory_map(str(tbar_source), "r") as mapped:
        reader = ipc.open_file(mapped)
        required = {"x", "y", "z"}
        missing = sorted(required - set(reader.schema.names))
        if missing:
            raise ValueError(f"T-bar source missing columns: {missing}")
        transmitters = tuple(
            name.removeprefix("nt_").removesuffix("_prob")
            for name in reader.schema.names
            if name.startswith("nt_") and name.endswith("_prob")
        )
        if not transmitters:
            raise ValueError("T-bar source requires at least one nt_*_prob column")
        probability_totals = {
            pair: dict.fromkeys(transmitters, 0.0) for pair in declared
        }
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            values = {
                transmitter: batch.column(f"nt_{transmitter}_prob").to_pylist()
                for transmitter in transmitters
            }
            for row_index, (x, y, z) in enumerate(
                zip(
                    batch.column("x").to_pylist(),
                    batch.column("y").to_pylist(),
                    batch.column("z").to_pylist(),
                    strict=True,
                )
            ):
                for pair in pairs_by_coordinate.get((int(x), int(y), int(z)), ()):
                    matched_counts[pair] += 1
                    for transmitter in transmitters:
                        probability_totals[pair][transmitter] += float(
                            values[transmitter][row_index]
                        )

    summaries = tuple(
        ContactTransmitterSummary(
            pre_id=pair[0],
            post_id=pair[1],
            contact_count=len(contact_coordinates[pair]),
            matched_tbar_count=matched_counts[pair],
            mean_probabilities={
                transmitter: total / matched_counts[pair]
                for transmitter, total in probability_totals[pair].items()
            }
            if matched_counts[pair]
            else {},
        )
        for pair in declared
    )
    contact_rows = sum(item.contact_count for item in summaries)
    matched_rows = sum(item.matched_tbar_count for item in summaries)
    return ContactTransmitterAudit(
        partner_contact_rows=contact_rows,
        matched_contact_rows=matched_rows,
        unmatched_contact_rows=contact_rows - matched_rows,
        edges=summaries,
    )
