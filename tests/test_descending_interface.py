import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.descending_interface import (
    DescendingDecoder,
    DescendingMap,
    population_silence_mask,
)
from flybrain.graph import EventConnectome

DN_MAP = DescendingMap((10, 11), (20, 21), (12,), (22,), (30, 31), (40, 41))


def graph() -> EventConnectome:
    ids = np.array([10, 11, 12, 20, 21, 22, 30, 31, 40, 41, 99], dtype=np.uint64)
    return EventConnectome(
        neuron_ids=ids,
        cell_types=("fixture",) * len(ids),
        roles=("interneuron",) * len(ids),
        transmitters=("acetylcholine",) * len(ids),
        superclasses=("descending_neuron",) * len(ids),
        outgoing=csr_array((len(ids), len(ids)), dtype=np.float32),
    )


def test_left_steering_is_ipsiversive_and_bilateral_activity_cancels() -> None:
    decoder = DescendingDecoder(DN_MAP, walking_drive=0.2)
    left = decoder.decode((10,))
    symmetric = decoder.decode((10, 11, 20, 21, 12, 22))
    assert left.command.turn > 0
    assert symmetric.command.turn == pytest.approx(0.0)


def test_bilateral_mdn_activity_reverses_without_turning() -> None:
    result = DescendingDecoder(DN_MAP, walking_drive=0.2).decode((30, 31, 40, 41))
    assert result.command.forward < 0
    assert result.command.turn == pytest.approx(0.0)


def test_mdn_laterality_adds_retreat_turn() -> None:
    result = DescendingDecoder(DN_MAP).decode((30, 31))
    assert result.command.forward < 0
    assert result.command.turn > 0


def test_silence_mask_selects_only_named_declared_ids() -> None:
    mask = population_silence_mask(graph(), DN_MAP, frozenset({"d_na02_left"}))
    np.testing.assert_array_equal(
        mask,
        [True, True, False, False, False, False, False, False, False, False, False],
    )


def test_silence_rejects_unknown_population() -> None:
    with pytest.raises(ValueError, match="unknown descending populations"):
        population_silence_mask(graph(), DN_MAP, frozenset({"not_a_population"}))


def test_map_rejects_overlapping_or_empty_populations() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        DescendingMap((1,), (1,), (2,), (3,), (4,), (5,))
    with pytest.raises(ValueError, match="positive IDs"):
        DescendingMap((), (1,), (2,), (3,), (4,), (5,))
