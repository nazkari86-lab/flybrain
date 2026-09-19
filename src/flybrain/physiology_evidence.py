"""Typed physiology evidence that is not silently generalized across cell types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EvidenceClass = Literal[
    "dataset_measurement",
    "model_assumption",
    "simulation_observation",
]


@dataclass(frozen=True)
class PhysiologyDatum:
    """One numerical value with explicit biological and epistemic provenance."""

    name: str
    value: float
    unit: str
    evidence_class: EvidenceClass
    source_doi: str
    population: str
    sample_size: int | None = None
    uncertainty: float | None = None
    uncertainty_kind: Literal["SD", "SEM"] | None = None
    source_detail: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.source_doi or not self.population:
            raise ValueError("physiology evidence identity must not be empty")
        if not math.isfinite(self.value):
            raise ValueError("physiology evidence values must be finite")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("physiology evidence sample size must be positive")
        if self.uncertainty is not None and (
            not math.isfinite(self.uncertainty) or self.uncertainty < 0
        ):
            raise ValueError("physiology evidence uncertainty must be finite and non-negative")
        if (self.uncertainty is None) != (self.uncertainty_kind is None):
            raise ValueError("uncertainty value and kind must be declared together")


def resolve_mbon_alpha3_ids(snapshot: Path) -> tuple[int, ...]:
    """Resolve exact MaleCNS MBON14 cells, the annotated MBON-alpha3 homologues."""

    table = pq.read_table(
        snapshot / "source-annotations.parquet",
        columns=["bodyId", "class", "type"],
    )
    body_ids = table.column("bodyId")
    if body_ids.null_count:
        raise ValueError("source annotation body IDs must not be null")
    values = [int(value) for value in body_ids.to_pylist()]
    if any(value <= 0 for value in values):
        raise ValueError("source annotation body IDs must be positive")
    if len(values) != len(set(values)):
        raise ValueError("source annotations contain duplicate body IDs")
    mask = pc.and_(
        pc.equal(table.column("class"), pa.scalar("MBON")),
        pc.equal(table.column("type"), pa.scalar("MBON14")),
    )
    selected = table.filter(pc.fill_null(mask, False)).column("bodyId").to_pylist()
    resolved = tuple(sorted(int(value) for value in selected))
    if not resolved:
        raise ValueError("source annotations require exact MBON14 cells")
    return resolved


def mbon_alpha3_evidence() -> tuple[PhysiologyDatum, ...]:
    """Return measurements and model fits for MBON-alpha3 without conflating them."""

    doi = "10.7554/eLife.77578"
    population = "MBON-alpha3 / MaleCNS MBON14"
    measured = (
        PhysiologyDatum(
            "main_resting_potential",
            -56.7,
            "mV",
            "dataset_measurement",
            doi,
            population,
            5,
            2.0,
            "SEM",
            "article Table 1, membrane-bound GFP cohort",
        ),
        PhysiologyDatum(
            "main_membrane_tau",
            16.06,
            "ms",
            "dataset_measurement",
            doi,
            population,
            5,
            2.20,
            "SEM",
            "article Table 1, membrane-bound GFP cohort",
        ),
        PhysiologyDatum(
            "main_membrane_capacitance",
            16.76,
            "pF",
            "dataset_measurement",
            doi,
            population,
            5,
            1.90,
            "SEM",
            "article Table 1, membrane-bound GFP cohort",
        ),
        PhysiologyDatum(
            "main_membrane_resistance",
            926.0,
            "MOhm",
            "dataset_measurement",
            doi,
            population,
            5,
            55.0,
            "SEM",
            "article electrophysiology results",
        ),
        PhysiologyDatum(
            "egfp_resting_potential",
            -60.6,
            "mV",
            "dataset_measurement",
            doi,
            population,
            4,
            1.3,
            "SEM",
            "elife-77578-table1-data1.xlsx, cytoplasmic EGFP cohort",
        ),
        PhysiologyDatum(
            "egfp_membrane_tau",
            14.48,
            "ms",
            "dataset_measurement",
            doi,
            population,
            4,
            1.57,
            "SEM",
            "elife-77578-table1-data1.xlsx, cytoplasmic EGFP cohort",
        ),
        PhysiologyDatum(
            "egfp_membrane_capacitance",
            13.95,
            "pF",
            "dataset_measurement",
            doi,
            population,
            4,
            2.69,
            "SEM",
            "elife-77578-table1-data1.xlsx, cytoplasmic EGFP cohort",
        ),
    )
    assumptions = (
        PhysiologyDatum(
            "fitted_axial_resistivity",
            85.41,
            "Ohm*cm",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, morphology-aware fit",
        ),
        PhysiologyDatum(
            "fitted_specific_capacitance",
            0.6961,
            "uF/cm^2",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, morphology-aware fit",
        ),
        PhysiologyDatum(
            "fitted_passive_conductance",
            9.399e-6,
            "S/cm^2",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, morphology-aware fit",
        ),
        PhysiologyDatum(
            "fitted_leak_reversal",
            -55.64,
            "mV",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, morphology-aware fit",
        ),
        PhysiologyDatum(
            "assumed_cholinergic_time_to_peak",
            0.44,
            "ms",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, transferred from Su and O'Dowd 2003",
        ),
        PhysiologyDatum(
            "assumed_cholinergic_reversal",
            8.9,
            "mV",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, transferred from Su and O'Dowd 2003",
        ),
        PhysiologyDatum(
            "fitted_synaptic_gmax",
            1.5627e-11,
            "S",
            "model_assumption",
            doi,
            population,
            source_detail="article Table 2, fitted to KC-to-MBON response target",
        ),
    )
    return measured + assumptions


def leg_motor_evidence() -> tuple[PhysiologyDatum, ...]:
    """Return measured adult front-leg tibia-flexor input resistance by motor type."""

    doi = "10.7554/eLife.56754"
    return tuple(
        PhysiologyDatum(
            name=f"{motor_type}_tibia_flexor_input_resistance",
            value=value,
            unit="MOhm",
            evidence_class="dataset_measurement",
            source_doi=doi,
            population=f"adult prothoracic {motor_type} tibia flexor motor neuron",
            sample_size=sample_size,
            source_detail="article Figure 3 and electrophysiology results",
        )
        for motor_type, value, sample_size in (
            ("fast", 150.0, 15),
            ("intermediate", 300.0, 11),
            ("slow", 700.0, 14),
        )
    )
