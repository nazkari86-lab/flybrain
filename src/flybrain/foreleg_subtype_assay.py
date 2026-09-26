"""Open-loop MaleCNS foreleg sensory-subtype to tibia-motor diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq

from flybrain.graph import EventConnectome
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.hexapod_neural_protocols import (
    _graph_digest,
    _run_proprio_condition,
    _validate_proprio_interfaces,
)
from flybrain.proprioceptive_interface import ProprioceptiveBank, ProprioceptiveMap
from flybrain.shiu import ShiuParameters


def load_foreleg_subtype_labels(
    annotations_path: Path, proprio: ProprioceptiveMap
) -> dict[int, str]:
    """Resolve exact foreleg bank IDs to retained MaleCNS sensory subclasses."""

    expected_side = {
        neuron_id: side
        for side, bank in (
            ("L", proprio.bank("left_fore")),
            ("R", proprio.bank("right_fore")),
        )
        for neuron_id in bank.neuron_ids
    }
    columns = (
        "bodyId",
        "subclass",
        "superclass",
        "class",
        "entryNerve",
        "rootSide",
    )
    rows = pq.read_table(annotations_path, columns=list(columns)).to_pylist()
    labels: dict[int, str] = {}
    for row in rows:
        neuron_id = row["bodyId"]
        if neuron_id not in expected_side:
            continue
        if neuron_id in labels:
            raise ValueError(f"duplicate foreleg annotation: {neuron_id}")
        if (
            row["superclass"] != "vnc_sensory"
            or row["class"] != "mechanosensory_proprioceptive"
            or row["entryNerve"] != "ProLN"
            or row["rootSide"] != expected_side[neuron_id]
            or not isinstance(row["subclass"], str)
            or not row["subclass"].strip()
        ):
            raise ValueError(f"foreleg annotation mismatch: {neuron_id}")
        labels[neuron_id] = row["subclass"]
    missing = sorted(expected_side.keys() - labels.keys())
    if missing:
        raise ValueError(f"missing foreleg subtype labels: {missing}")
    return labels


def run_foreleg_subtype_assay(
    graph: EventConnectome,
    proprio: ProprioceptiveMap,
    motor: HexapodMotorMap,
    *,
    subtype_by_id: Mapping[int, str],
    steps: int,
    seed: int,
    drive_interval_steps: int = 25,
    drive_amplitude_mv: float = 10.0,
) -> dict[str, object]:
    """Pulse annotated foreleg subsets, then lesion and exactly replay each.

    Fixed per-cell pulses are a diagnostic intervention, not an observed
    proprioceptive transfer function. Different-sized subsets are not a
    size-controlled efficacy comparison; the output preserves their counts.
    """

    if type(steps) is not int or steps <= 0:
        raise ValueError("subtype assay steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("subtype assay seed must be a non-negative integer")
    _validate_proprio_interfaces(graph, proprio, motor)
    fore_banks = (proprio.bank("left_fore"), proprio.bank("right_fore"))
    fore_ids = {neuron_id for bank in fore_banks for neuron_id in bank.neuron_ids}
    missing = sorted(fore_ids - subtype_by_id.keys())
    if missing:
        raise ValueError(f"missing foreleg subtype labels: {missing}")
    if any(
        not isinstance(subtype_by_id[neuron_id], str) or not subtype_by_id[neuron_id].strip()
        for neuron_id in fore_ids
    ):
        raise ValueError("foreleg subtype labels must be nonempty strings")

    before_digest = _graph_digest(graph)
    params = ShiuParameters()
    conditions: list[dict[str, object]] = []
    leave_one_out_controls: list[dict[str, object]] = []
    subtype_counts: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    tibia_names = tuple(
        f"{side}_fore_tibia_{direction}"
        for side in ("left", "right")
        for direction in ("flexor", "extensor")
    )

    def run_condition(
        side: str,
        bank: ProprioceptiveBank,
        selected_map: ProprioceptiveMap,
        subtype: str,
    ) -> dict[str, object]:
        run = partial(
            _run_proprio_condition,
            graph,
            selected_map,
            motor,
            input_bank=bank.name,
            silence_all_motor=False,
            steps=steps,
            seed=seed,
            params=params,
            interval_steps=drive_interval_steps,
            amplitude_mv=drive_amplitude_mv,
        )
        normal = run(name="normal", silenced_banks=frozenset())
        lesion = run(name="source_lesion", silenced_banks=frozenset({bank.name}))
        replay = run(name="replay", silenced_banks=frozenset())
        return {
            "side": side,
            "subtype": subtype,
            "source_ids": normal.event_target_ids,
            "source_neuron_count": len(normal.event_target_ids),
            "source_spikes": normal.input_spikes,
            "source_lesion_source_spikes": lesion.input_spikes,
            "motor_spikes": normal.motor_spikes,
            "motor_spikes_by_population": normal.motor_spikes_by_population,
            "source_lesion_motor_spikes": lesion.motor_spikes,
            "tibia_spikes": {name: normal.motor_spikes_by_population[name] for name in tibia_names},
            "source_lesion_tibia_spikes": {
                name: lesion.motor_spikes_by_population[name] for name in tibia_names
            },
            "source_lesion_same_events": lesion.event_digest == normal.event_digest,
            "replay_exact": replay.trace_digest == normal.trace_digest,
            "event_digest": normal.event_digest,
            "trace_digest": normal.trace_digest,
        }

    for side, bank in zip(("left", "right"), fore_banks, strict=True):
        for subtype in sorted({subtype_by_id[value] for value in bank.neuron_ids}):
            selected_ids = tuple(
                value for value in bank.neuron_ids if subtype_by_id[value] == subtype
            )
            subtype_counts[side][subtype] = len(selected_ids)
            selected_bank = ProprioceptiveBank(
                name=bank.name, leg=bank.leg, neuron_ids=selected_ids
            )
            selected_map = ProprioceptiveMap(
                banks=tuple(
                    selected_bank if item.name == bank.name else item for item in proprio.banks
                )
            )

            conditions.append(run_condition(side, bank, selected_map, subtype))
            remaining_ids = tuple(
                value for value in bank.neuron_ids if subtype_by_id[value] != subtype
            )
            if remaining_ids:
                remaining_bank = ProprioceptiveBank(
                    name=bank.name, leg=bank.leg, neuron_ids=remaining_ids
                )
                remaining_map = ProprioceptiveMap(
                    banks=tuple(
                        remaining_bank if item.name == bank.name else item for item in proprio.banks
                    )
                )
                leave_one_out_controls.append(
                    {
                        **run_condition(side, bank, remaining_map, "all_except"),
                        "excluded_subtype": subtype,
                    }
                )

    whole_bank_controls = tuple(
        run_condition(side, bank, proprio, "all")
        for side, bank in zip(("left", "right"), fore_banks, strict=True)
    )

    paired = sorted(subtype_counts["left"].keys() & subtype_counts["right"].keys())
    unpaired = sorted(subtype_counts["left"].keys() ^ subtype_counts["right"].keys())
    return {
        "protocol": "foreleg-proprio-subtype-open-loop-v1",
        "evidence_kind": "simulation_observation",
        "closed_loop": False,
        "steps": steps,
        "seed": seed,
        "drive_interval_steps": drive_interval_steps,
        "drive_amplitude_mv": drive_amplitude_mv,
        "conditions": tuple(conditions),
        "whole_bank_controls": whole_bank_controls,
        "leave_one_out_controls": tuple(leave_one_out_controls),
        "mirror_pairs": tuple(
            {
                "subtype": subtype,
                "left_count": subtype_counts["left"][subtype],
                "right_count": subtype_counts["right"][subtype],
            }
            for subtype in paired
        ),
        "unpaired_subtypes": tuple(unpaired),
        "graph_unchanged": _graph_digest(graph) == before_digest,
        "behavioral_claim_allowed": False,
    }
