import numpy as np
from scipy.sparse import csr_array

from flybrain.graph import EventConnectome
from flybrain.retinal_interface import VisualLoomingMap
from flybrain.visual_looming_assay import run_visual_looming_assay


def fixture() -> tuple[EventConnectome, VisualLoomingMap]:
    ids = np.arange(1, 9, dtype=np.uint64)
    rows = (0, 1, 2, 3, 0, 1, 2, 3)
    cols = (4, 5, 4, 5, 6, 7, 6, 7)
    graph = EventConnectome(
        neuron_ids=ids,
        cell_types=("visual",) * 4 + ("descending",) * 4,
        roles=("sensory",) * 4 + ("descending",) * 4,
        transmitters=("acetylcholine",) * 8,
        superclasses=("visual_projection",) * 4 + ("descending_neuron",) * 4,
        outgoing=csr_array(
            (np.full(len(rows), 100.0, dtype=np.float32), (rows, cols)),
            shape=(8, 8),
        ),
    )
    mapping = VisualLoomingMap(
        left_lc4_ids=(1,),
        right_lc4_ids=(2,),
        left_lplc2_ids=(3,),
        right_lplc2_ids=(4,),
        left_dnp01_ids=(5,),
        right_dnp01_ids=(6,),
        left_dnp02_ids=(7,),
        right_dnp02_ids=(8,),
    )
    return graph, mapping


def test_looming_path_is_causal_replayable_and_not_a_behavior_claim() -> None:
    graph, mapping = fixture()
    result = run_visual_looming_assay(graph, mapping, steps=30, seed=7)

    normal = result.conditions["looming"]
    static = result.conditions["static"]
    joint_lesion = result.conditions["lc4_lplc2_lesion"]
    assert normal.dnp01_spikes > static.dnp01_spikes
    assert normal.dnp01_spikes > joint_lesion.dnp01_spikes
    assert normal.dnp02_spikes > 0
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.behavioral_claim_allowed is False
