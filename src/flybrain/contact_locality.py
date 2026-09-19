"""Measure KC and DAN contact locality on exact retained MaleCNS synapses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MbonContactLocality:
    """Nearest-signal distances for all declared target contacts on one MBON."""

    mbon_id: int
    target_contact_count: int
    signal_contact_count: int
    nearest_signal_median: float
    nearest_signal_p90: float


@dataclass(frozen=True)
class ContactLocalityAudit:
    """Geometry measurement, deliberately separate from a learning-rule assumption."""

    target_contact_rows: int
    signal_contact_rows: int
    target_mbon_ids_without_signal_contacts: tuple[int, ...]
    mbon_summaries: tuple[MbonContactLocality, ...]


def _validate_pairs(name: str, pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    declared = tuple(sorted(set(pairs)))
    if not declared or any(
        type(pre_id) is not int or type(post_id) is not int or pre_id <= 0 or post_id <= 0
        for pre_id, post_id in declared
    ):
        raise ValueError(f"{name} must contain positive integer endpoints")
    if len(declared) != len(pairs):
        raise ValueError(f"{name} must be unique")
    return declared


def audit_contact_locality(
    source: Path,
    *,
    target_pairs: tuple[tuple[int, int], ...],
    signal_pairs: tuple[tuple[int, int], ...],
) -> ContactLocalityAudit:
    """Measure nearest signal contacts for declared target contacts on each MBON.

    Coordinates are returned in the source's native MaleCNS coordinate units.
    This function neither selects a locality radius nor changes any weights.
    """

    targets = _validate_pairs("target_pairs", target_pairs)
    signals = _validate_pairs("signal_pairs", signal_pairs)
    target_set = frozenset(targets)
    signal_set = frozenset(signals)
    if target_set & signal_set:
        raise ValueError("target_pairs and signal_pairs must not overlap")
    pre_values = pa.array(
        sorted({pre_id for pre_id, _ in target_set | signal_set}), type=pa.uint64()
    )
    target_by_post: dict[int, list[tuple[int, int, int]]] = {}
    signal_by_post: dict[int, list[tuple[int, int, int]]] = {}

    with pa.memory_map(str(source), "r") as mapped:
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
                coordinate = (int(x), int(y), int(z))
                if pair in target_set:
                    target_by_post.setdefault(pair[1], []).append(coordinate)
                elif pair in signal_set:
                    signal_by_post.setdefault(pair[1], []).append(coordinate)

    summaries = []
    for mbon_id in sorted(target_by_post.keys() & signal_by_post.keys()):
        targets_for_mbon = np.asarray(target_by_post[mbon_id], dtype=np.float64)
        signals_for_mbon = np.asarray(signal_by_post[mbon_id], dtype=np.float64)
        distances = cKDTree(signals_for_mbon).query(targets_for_mbon, k=1)[0]
        summaries.append(
            MbonContactLocality(
                mbon_id=mbon_id,
                target_contact_count=targets_for_mbon.shape[0],
                signal_contact_count=signals_for_mbon.shape[0],
                nearest_signal_median=float(np.median(distances)),
                nearest_signal_p90=float(np.quantile(distances, 0.9)),
            )
        )
    return ContactLocalityAudit(
        target_contact_rows=sum(len(items) for items in target_by_post.values()),
        signal_contact_rows=sum(len(items) for items in signal_by_post.values()),
        target_mbon_ids_without_signal_contacts=tuple(
            sorted(target_by_post.keys() - signal_by_post.keys())
        ),
        mbon_summaries=tuple(summaries),
    )
