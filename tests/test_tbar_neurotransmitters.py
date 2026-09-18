from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flybrain.tbar_neurotransmitters import (
    audit_body_neurotransmitters,
    audit_contact_transmitters,
    audit_tbar_neurotransmitters,
)


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


def test_body_audit_retains_official_consensus_over_raw_prediction(tmp_path: Path) -> None:
    source = tmp_path / "body-neurotransmitters.feather"
    feather.write_feather(
        pa.table(
            {
                "body": [7, 9],
                "predicted_nt": ["dopamine", "gaba"],
                "predicted_nt_confidence": [0.6, 0.9],
                "consensus_nt": ["acetylcholine", "gaba"],
            }
        ),
        source,
    )

    audit = audit_body_neurotransmitters(source, body_ids=(7, 9, 42))

    assert audit.source_body_rows == 2
    assert audit.missing_body_ids == (42,)
    assert audit.consensus_prediction_disagreements == 1
    assert audit.bodies[0].body_id == 7
    assert audit.bodies[0].consensus_transmitter == "acetylcholine"
    assert audit.bodies[0].predicted_transmitter == "dopamine"
    assert audit.bodies[0].prediction_confidence == pytest.approx(0.6)


def test_contact_audit_joins_tbars_to_exact_presynaptic_contacts(tmp_path: Path) -> None:
    tbars = tmp_path / "tbars.feather"
    partners = tmp_path / "partners.feather"
    feather.write_feather(
        pa.table(
            {
                "x": [1, 2, 3],
                "y": [1, 2, 3],
                "z": [1, 2, 3],
                "body": [7, 7, 9],
                "nt_acetylcholine_prob": [0.8, 0.4, 0.1],
                "nt_dopamine_prob": [0.2, 0.6, 0.9],
            }
        ),
        tbars,
        chunksize=1,
    )
    feather.write_feather(
        pa.table(
            {
                "x_pre": [1, 2, 3, 99],
                "y_pre": [1, 2, 3, 99],
                "z_pre": [1, 2, 3, 99],
                "body_pre": [7, 7, 9, 7],
                "body_post": [20, 20, 20, 20],
            }
        ),
        partners,
        chunksize=1,
    )

    audit = audit_contact_transmitters(
        tbars,
        partners,
        edge_pairs=((7, 20), (9, 20)),
    )

    assert audit.partner_contact_rows == 4
    assert audit.matched_contact_rows == 3
    assert audit.unmatched_contact_rows == 1
    assert [(item.pre_id, item.post_id, item.contact_count) for item in audit.edges] == [
        (7, 20, 3),
        (9, 20, 1),
    ]
    assert audit.edges[0].matched_tbar_count == 2
    assert audit.edges[0].mean_probabilities == pytest.approx(
        {"acetylcholine": 0.6, "dopamine": 0.4}
    )
