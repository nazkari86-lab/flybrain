from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flybrain.hexapod_body import LEG_NAMES
from flybrain.proprioceptive_interface import ProprioceptiveBank, ProprioceptiveMap
from flybrain.proprioceptive_subtypes import load_proprioceptive_subtypes


def tiny_map() -> ProprioceptiveMap:
    return ProprioceptiveMap(
        banks=tuple(
            ProprioceptiveBank(name=f"{leg}_proprioception", leg=leg, neuron_ids=(index + 1,))
            for index, leg in enumerate(LEG_NAMES)
        )
    )


def annotations(path: Path, *, wrong_side: bool = False) -> None:
    pq.write_table(
        pa.table(
            {
                "bodyId": list(range(1, 7)),
                "subclass": [
                    "chordotonal organ",
                    "campaniform sensilla",
                    "hair plate",
                    "leg",
                    "leg",
                    "leg",
                ],
                "superclass": ["vnc_sensory"] * 6,
                "class": ["mechanosensory_proprioceptive"] * 6,
                "entryNerve": ["ProLN", "ProLN", "MesoLN", "MesoLN", "MetaLN", "MetaLN"],
                "rootSide": ["R" if wrong_side else "L", "R", "L", "R", "L", "R"],
            }
        ),
        path,
    )


def test_loader_requires_exact_leg_anatomy(tmp_path: Path) -> None:
    path = tmp_path / "annotations.parquet"
    annotations(path)

    labels = load_proprioceptive_subtypes(path, tiny_map())

    assert labels == {
        1: "chordotonal organ",
        2: "campaniform sensilla",
        3: "hair plate",
        4: "leg",
        5: "leg",
        6: "leg",
    }
    annotations(path, wrong_side=True)
    with pytest.raises(ValueError, match="annotation mismatch"):
        load_proprioceptive_subtypes(path, tiny_map())


def test_loader_rejects_missing_sensor(tmp_path: Path) -> None:
    path = tmp_path / "annotations.parquet"
    annotations(path)
    pq.write_table(pq.read_table(path).slice(0, 5), path)
    with pytest.raises(ValueError, match="missing"):
        load_proprioceptive_subtypes(path, tiny_map())
