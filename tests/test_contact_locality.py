from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flybrain.contact_locality import audit_contact_locality


def test_locality_audit_measures_nearest_signal_contact_per_target_contact(tmp_path: Path) -> None:
    source = tmp_path / "partners.feather"
    feather.write_feather(
        pa.table(
            {
                "x_pre": [0, 3, 0, 99],
                "y_pre": [0, 4, 4, 99],
                "z_pre": [0, 0, 0, 99],
                "body_pre": [10, 10, 30, 999],
                "body_post": [20, 20, 20, 20],
            }
        ),
        source,
        chunksize=1,
    )

    audit = audit_contact_locality(
        source,
        target_pairs=((10, 20),),
        signal_pairs=((30, 20),),
    )

    assert audit.target_contact_rows == 2
    assert audit.signal_contact_rows == 1
    assert len(audit.mbon_summaries) == 1
    summary = audit.mbon_summaries[0]
    assert summary.mbon_id == 20
    assert summary.target_contact_count == 2
    assert summary.signal_contact_count == 1
    assert summary.nearest_signal_median == pytest.approx(3.5)
    assert summary.nearest_signal_p90 == pytest.approx(3.9)
