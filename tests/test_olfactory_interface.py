import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.olfactory_interface import OlfactoryReceptorMap


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
