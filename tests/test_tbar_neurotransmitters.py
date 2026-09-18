from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flybrain.tbar_neurotransmitters import audit_tbar_neurotransmitters


def test_audit_aggregates_only_requested_presynaptic_bodies(tmp_path: Path) -> None:
    source = tmp_path / "tbars.feather"
    feather.write_feather(
        pa.table(
            {
                "point_id": [10, 11, 12, 13],
                "body": [7, 7, 9, 11],
                "nt_acetylcholine_prob": [0.8, 0.6, 0.1, 0.3],
                "nt_dopamine_prob": [0.1, 0.2, 0.2, 0.1],
                "nt_gaba_prob": [0.1, 0.2, 0.7, 0.6],
            }
        ),
        source,
        chunksize=2,
    )

    audit = audit_tbar_neurotransmitters(source, body_ids=(7, 9, 42))

    assert audit.source_tbar_rows == 4
    assert audit.matched_tbar_rows == 3
    assert audit.missing_body_ids == (42,)
    assert [(summary.body_id, summary.tbar_count) for summary in audit.bodies] == [
        (7, 2),
        (9, 1),
    ]
    assert audit.bodies[0].mean_probabilities == pytest.approx({
        "acetylcholine": 0.7,
        "dopamine": 0.15,
        "gaba": 0.15,
    })
    assert audit.bodies[1].dominant_transmitter == "gaba"
