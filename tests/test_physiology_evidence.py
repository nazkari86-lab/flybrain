from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from flybrain.physiology_evidence import (
    leg_motor_evidence,
    mbon_alpha3_evidence,
    resolve_mbon_alpha3_ids,
)


def test_mbon_alpha3_resolver_binds_only_exact_mbon14_type(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(
        pa.table(
            {
                "bodyId": [10, 11, 12, 13],
                "class": ["MBON", "MBON", "MBON", None],
                "type": ["MBON14", "MBON13", "MBON14-like", "MBON14"],
            }
        ),
        snapshot / "source-annotations.parquet",
    )

    assert resolve_mbon_alpha3_ids(snapshot) == (10,)


def test_mbon_alpha3_evidence_keeps_measurements_and_fits_separate() -> None:
    evidence = {datum.name: datum for datum in mbon_alpha3_evidence()}

    assert evidence["main_membrane_tau"].value == 16.06
    assert evidence["main_membrane_tau"].unit == "ms"
    assert evidence["main_membrane_tau"].evidence_class == "dataset_measurement"
    assert evidence["main_membrane_tau"].sample_size == 5
    assert evidence["egfp_membrane_tau"].value == 14.48
    assert evidence["egfp_membrane_tau"].sample_size == 4
    assert evidence["fitted_specific_capacitance"].value == 0.6961
    assert evidence["fitted_specific_capacitance"].evidence_class == "model_assumption"
    assert evidence["fitted_synaptic_gmax"].value == 1.5627e-11
    assert evidence["fitted_synaptic_gmax"].unit == "S"


def test_leg_motor_resistance_measurements_remain_type_specific() -> None:
    evidence = {datum.name: datum for datum in leg_motor_evidence()}

    assert evidence["fast_tibia_flexor_input_resistance"].value == 150.0
    assert evidence["intermediate_tibia_flexor_input_resistance"].value == 300.0
    assert evidence["slow_tibia_flexor_input_resistance"].value == 700.0
    assert all(
        datum.evidence_class == "dataset_measurement" for datum in evidence.values()
    )
