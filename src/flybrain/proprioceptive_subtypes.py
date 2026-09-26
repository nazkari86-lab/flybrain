"""Resolve retained MaleCNS leg-sensory subtypes without inferred anatomy."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from flybrain.proprioceptive_interface import ProprioceptiveMap

_NERVE_BY_SEGMENT = {"fore": "ProLN", "middle": "MesoLN", "hind": "MetaLN"}
_SUBTYPES = frozenset({"chordotonal organ", "campaniform sensilla", "hair plate", "leg"})


def load_proprioceptive_subtypes(
    annotations_path: Path, mapping: ProprioceptiveMap
) -> dict[int, str]:
    """Require one matching source annotation for every registered leg sensor."""

    expected = {
        neuron_id: (
            "L" if bank.leg.startswith("left_") else "R",
            _NERVE_BY_SEGMENT[bank.leg.split("_")[1]],
        )
        for bank in mapping.banks
        for neuron_id in bank.neuron_ids
    }
    rows = pq.read_table(
        annotations_path,
        columns=["bodyId", "subclass", "superclass", "class", "entryNerve", "rootSide"],
    ).to_pylist()
    labels: dict[int, str] = {}
    for row in rows:
        neuron_id = row["bodyId"]
        if neuron_id not in expected:
            continue
        if neuron_id in labels:
            raise ValueError(f"duplicate proprioceptive annotation: {neuron_id}")
        side, nerve = expected[neuron_id]
        if (
            row["superclass"] != "vnc_sensory"
            or row["class"] != "mechanosensory_proprioceptive"
            or row["entryNerve"] != nerve
            or row["rootSide"] != side
            or row["subclass"] not in _SUBTYPES
        ):
            raise ValueError(f"proprioceptive annotation mismatch: {neuron_id}")
        labels[neuron_id] = row["subclass"]
    missing = sorted(expected.keys() - labels.keys())
    if missing:
        raise ValueError(f"missing proprioceptive annotations: {missing}")
    return labels
