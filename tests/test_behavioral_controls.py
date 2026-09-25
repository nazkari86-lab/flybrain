import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_controls import CONDITIONS, ControlCondition, build_condition
from flybrain.graph import EventConnectome
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay


def binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1, 3, 5], dtype=np.int64), canonical_edge_count=6
        ),
        pre_ids=np.array([10, 11, 12], dtype=np.uint64),
        post_ids=np.array([20, 21, 22], dtype=np.uint64),
    )


def learning() -> AssociativeCalibrationConfig:
    return AssociativeCalibrationConfig(
        cue_ids=(10, 11, 12),
        mbon_ids=(20, 21, 22),
        dan_to_mbon_pairs=((30, 20), (31, 21)),
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_conditions_are_independent_and_explicit(condition: ControlCondition) -> None:
    original = binding()
    result = build_condition(condition, original, learning(), seed=7)
    result.overlay.multipliers[0] = 0.25

    assert original.overlay.multipliers[0] == 1.0
    assert result.condition == condition
    assert result.rewired is (condition == "rewired_control")
    if condition == "rewired_control":
        assert result.post_ids == (20, 21, 22)


def test_controls_have_expected_effects() -> None:
    original = binding()
    no_plasticity = build_condition("no_plasticity", original, learning(), seed=7)
    kc_lesion = build_condition("kc_mbon_lesion", original, learning(), seed=7)
    dan_lesion = build_condition("dan_lesion", original, learning(), seed=7)
    assert no_plasticity.overlay.multipliers.tolist() == [1.0, 1.0, 1.0]
    assert kc_lesion.overlay.multipliers.tolist() == [0.0, 0.0, 0.0]
    assert dan_lesion.dan_enabled is False


def test_controls_declare_whether_their_overlay_may_learn() -> None:
    original = binding()

    normal = build_condition("normal", original, learning(), seed=7)
    no_plasticity = build_condition("no_plasticity", original, learning(), seed=7)
    dan_lesion = build_condition("dan_lesion", original, learning(), seed=7)
    kc_lesion = build_condition("kc_mbon_lesion", original, learning(), seed=7)
    route_permutation = build_condition("rewired_control", original, learning(), seed=7)

    assert normal.plasticity_enabled is True
    assert no_plasticity.plasticity_enabled is False
    assert dan_lesion.plasticity_enabled is True
    assert kc_lesion.plasticity_enabled is False
    assert route_permutation.plasticity_enabled is True


def test_rewired_control_changes_real_edges_without_mutating_canonical_graph() -> None:
    from flybrain.behavioral_controls import rewire_plastic_edges

    graph = EventConnectome(
        neuron_ids=np.array([10, 11, 20, 21], dtype=np.uint64),
        cell_types=("KC", "KC", "MBON", "MBON"),
        roles=("learning_kc", "learning_kc", "learning_mbon", "learning_mbon"),
        transmitters=("acetylcholine",) * 4,
        superclasses=("fixture",) * 4,
        outgoing=csr_array(
            (np.array([2.0, 3.0], dtype=np.float32),
             (np.array([0, 1]), np.array([2, 3]))),
            shape=(4, 4),
        ),
    )
    binding = PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([0, 1], dtype=np.int64), canonical_edge_count=2
        ),
        pre_ids=np.array([10, 11], dtype=np.uint64),
        post_ids=np.array([20, 21], dtype=np.uint64),
    )
    canonical_indices = graph.outgoing.indices.copy()
    original_indegree = np.bincount(graph.outgoing.indices, minlength=4)

    result = rewire_plastic_edges(graph, binding, seed=7)

    assert result.changed_edges == 2
    assert result.graph.outgoing.indices.tolist() == [3, 2]
    assert result.binding.post_ids.tolist() == [21, 20]
    np.testing.assert_array_equal(graph.outgoing.indices, canonical_indices)
    np.testing.assert_array_equal(result.graph.outgoing.data, graph.outgoing.data)
    np.testing.assert_array_equal(result.graph.outgoing.indptr, graph.outgoing.indptr)
    np.testing.assert_array_equal(
        np.bincount(result.graph.outgoing.indices, minlength=4), original_indegree
    )


def test_rewired_control_rejects_swaps_that_duplicate_nonplastic_edges() -> None:
    from flybrain.behavioral_controls import rewire_plastic_edges

    graph = EventConnectome(
        neuron_ids=np.array([10, 11, 20, 21], dtype=np.uint64),
        cell_types=("KC", "KC", "MBON", "MBON"),
        roles=("learning_kc", "learning_kc", "learning_mbon", "learning_mbon"),
        transmitters=("acetylcholine",) * 4,
        superclasses=("fixture",) * 4,
        outgoing=csr_array(
            (
                np.ones(4, dtype=np.float32),
                (np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])),
            ),
            shape=(4, 4),
        ),
    )
    binding = PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([0, 3], dtype=np.int64), canonical_edge_count=4
        ),
        pre_ids=np.array([10, 11], dtype=np.uint64),
        post_ids=np.array([20, 21], dtype=np.uint64),
    )

    result = rewire_plastic_edges(graph, binding, seed=7)

    assert result.changed_edges == 0
    assert result.graph is graph
