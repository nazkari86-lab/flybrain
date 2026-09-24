import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome
from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS, HexapodMotorMap, MotorGroup
from flybrain.mbon_descending_assay import (
    audit_mbon_descending_pathways,
    run_mbon_descending_causal_assay,
    run_retained_mbon_descending_assay,
)


def _motor_map() -> HexapodMotorMap:
    return HexapodMotorMap(
        groups=tuple(
            MotorGroup(
                name=name,
                leg=leg,
                joint=joint,
                direction=direction,
                neuron_ids=(100 + index,),
            )
            for index, (name, leg, joint, direction) in enumerate(CANONICAL_MOTOR_GROUPS)
        )
    )


def _descending_map() -> DescendingMap:
    return DescendingMap(
        d_na02_left=(20,),
        d_na02_right=(21,),
        d_ng13_left=(22,),
        d_ng13_right=(23,),
        mdn_left=(24,),
        mdn_right=(25,),
    )


def _graph() -> EventConnectome:
    ids = np.asarray([10, 11, 20, 21, 22, 23, 24, 25, 50, *range(100, 136)], dtype=np.uint64)
    index = {int(value): position for position, value in enumerate(ids)}
    edges = (
        (10, 20, 400.0),
        (20, 100, 400.0),
        (11, 50, 400.0),
    )
    return EventConnectome(
        neuron_ids=ids,
        cell_types=(
            "MBON-a",
            "MBON-b",
            "DNa02",
            "DNa02",
            "DNg13",
            "DNg13",
            "MDN",
            "MDN",
            "control",
            *("motor",) * 36,
        ),
        roles=(
            "learning_mbon",
            "learning_mbon",
            *("descending",) * 6,
            "interneuron",
            *("motor",) * 36,
        ),
        transmitters=("acetylcholine",) * len(ids),
        superclasses=(
            "central",
            "central",
            *("descending_neuron",) * 6,
            "central",
            *("motor_neuron",) * 36,
        ),
        outgoing=csr_array(
            (
                np.asarray([item[2] for item in edges], dtype=np.float32),
                (
                    [index[item[0]] for item in edges],
                    [index[item[1]] for item in edges],
                ),
            ),
            shape=(len(ids), len(ids)),
        ),
    )


def test_pathway_audit_separates_direct_signed_authority_from_matched_controls() -> None:
    result = audit_mbon_descending_pathways(
        _graph(),
        mbon_ids=(10, 11),
        approach_mbon_ids=(10,),
        descending=_descending_map(),
        motor=_motor_map(),
        max_hops=4,
    )

    assert result.learned_mbon_count == 2
    assert result.approach_mbon_count == 1
    assert result.avoidance_mbon_count == 1
    assert result.direct_selected_dn_edge_count == 1
    assert result.direct_selected_dn_sign_counts == {-1: 0, 0: 0, 1: 1}
    assert result.excitatory_direct_mbon_ids == (10,)
    assert result.matched_control_mbon_ids == (11,)
    assert result.profile(10).selected_dn_shortest_hops == 1
    assert result.profile(10).motor_shortest_hops == 2
    assert result.profile(11).authority_class == "disconnected"


def test_causal_assay_requires_lesion_restoration_replay_and_motor_specificity() -> None:
    graph = _graph()
    audit = audit_mbon_descending_pathways(
        graph,
        mbon_ids=(10, 11),
        approach_mbon_ids=(10,),
        descending=_descending_map(),
        motor=_motor_map(),
        max_hops=4,
    )

    result = run_mbon_descending_causal_assay(
        graph,
        audit=audit,
        descending=_descending_map(),
        motor=_motor_map(),
        steps=80,
        seed=7,
    )

    assert result.conditions["normal"].selected_descending_spikes > 0
    assert result.conditions["normal"].motor_spikes > 0
    assert result.conditions["normal"].decoded_torque_l1_nm_s > 0.0
    assert result.conditions["source_lesion"].selected_descending_spikes == 0
    assert result.conditions["source_lesion"].motor_spikes == 0
    assert result.conditions["source_lesion"].decoded_torque_l1_nm_s == 0.0
    assert result.conditions["descending_lesion"].motor_spikes == 0
    assert result.conditions["descending_lesion"].decoded_torque_l1_nm_s == 0.0
    assert result.conditions["matched_control"].motor_spikes == 0
    assert result.causal_mbon_to_descending_claim_allowed is True
    assert result.specific_mbon_dn_motor_claim_allowed is True
    assert result.autonomous_behavior_claim_allowed is False
    assert result.replay_exact is True
    assert result.restoration_exact is True
    assert result.graph_unchanged is True


def test_retained_malecns_assay_reports_causality_without_overclaiming_specificity() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")

    result = run_retained_mbon_descending_assay(
        Path(raw_snapshot),
        steps=500,
        seed=7,
    )

    assert result.graph_neurons == 166_606
    assert result.graph_edges == 6_240_402
    assert result.pathways.learned_mbon_count == 91
    assert result.pathways.approach_mbon_count == 49
    assert result.pathways.avoidance_mbon_count == 42
    assert result.pathways.direct_selected_dn_edge_count == 16
    assert result.pathways.direct_selected_dn_sign_counts == {-1: 10, 0: 0, 1: 6}
    assert result.pathways.excitatory_direct_mbon_ids == (
        10599,
        508595,
        519128,
        524893,
    )
    assert result.causal.causal_mbon_to_descending_claim_allowed is True
    assert result.causal.specific_mbon_dn_motor_claim_allowed is False
    assert result.autonomous_behavior_claim_allowed is False
