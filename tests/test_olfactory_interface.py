import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.olfactory_interface import OlfactoryReceptorMap, task_odor_assignment


def test_task_odors_use_causally_validated_food_and_harm_channels() -> None:
    assignment = task_odor_assignment()
    assert assignment.food_cell_types == ("ORN_DM1", "ORN_VA2")
    assert assignment.threat_cell_types == ("ORN_DA2",)
    assert assignment.food_evidence_doi == "10.1038/nature07983"
    assert assignment.threat_evidence_doi == "10.1016/j.cell.2012.09.046"
    assert assignment.evidence_scope == "adult_innate_valence"
    assert set(assignment.food_cell_types).isdisjoint(assignment.threat_cell_types)


def graph() -> EventConnectome:
    return EventConnectome(
        neuron_ids=np.array([1, 2, 3], dtype=np.uint64),
        cell_types=("ORN_DA1", "ORN_DA1", "ORN_VA1d"),
        roles=("fixture",) * 3,
        transmitters=("acetylcholine",) * 3,
        superclasses=("fixture",) * 3,
        outgoing=csr_array((3, 3), dtype=np.float32),
    )


def test_resolves_exact_orn_type_banks_and_channel_unions() -> None:
    mapping = OlfactoryReceptorMap.from_graph(graph(), olfactory_neuron_ids=(1, 2, 3))

    assert [bank.cell_type for bank in mapping.banks] == ["ORN_DA1", "ORN_VA1d"]
    assert mapping.channel_ids(("ORN_VA1d", "ORN_DA1")) == (1, 2, 3)
    assert mapping.channel_ids(("ORN_DA1",)) == (1, 2)


def test_rejects_unknown_or_untyped_receptor_channels() -> None:
    mapping = OlfactoryReceptorMap.from_graph(graph(), olfactory_neuron_ids=(1, 2, 3))

    with pytest.raises(ValueError, match="unknown ORN"):
        mapping.channel_ids(("ORN_DM1",))


def test_excludes_untyped_neurons_without_mixing_them_into_a_channel() -> None:
    untyped = EventConnectome(
        neuron_ids=np.array([1, 2], dtype=np.uint64),
        cell_types=("ORN_DA1", "untyped"),
        roles=("fixture", "fixture"),
        transmitters=("acetylcholine", "acetylcholine"),
        superclasses=("fixture", "fixture"),
        outgoing=csr_array((2, 2), dtype=np.float32),
    )

    mapping = OlfactoryReceptorMap.from_graph(untyped, olfactory_neuron_ids=(1, 2))

    assert mapping.channel_ids(("ORN_DA1",)) == (1,)
    assert mapping.excluded_untyped_neuron_ids == (2,)


def test_side_channel_uses_measured_left_right_ids_only() -> None:
    mapping = OlfactoryReceptorMap(
        banks=(
            {
                "cell_type": "ORN_DA1",
                "neuron_ids": (1, 2, 3),
                "left_ids": (1,),
                "right_ids": (2,),
                "unknown_side_ids": (3,),
            },
        )
    )

    assert mapping.side_channel_ids(("ORN_DA1",), "L") == (1,)
    assert mapping.side_channel_ids(("ORN_DA1",), "R") == (2,)
