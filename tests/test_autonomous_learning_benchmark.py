import numpy as np
import pytest
from pydantic import ValidationError
from scipy.sparse import csr_array

from flybrain.autonomous_learning_benchmark import (
    AssociativeCalibrationConfig,
    run_associative_calibration,
)
from flybrain.conditioning_world import ConditioningSchedule
from flybrain.graph import EventConnectome
from flybrain.mushroom_body_learning import MushroomBodyLearningParameters
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters


def synthetic_calibration_graph() -> EventConnectome:
    return EventConnectome(
        neuron_ids=np.array([10, 20, 30], dtype=np.uint64),
        cell_types=("Kenyon_Cell", "MBON", "DAN"),
        roles=("learning_kc", "learning_mbon", "dan_appetitive"),
        transmitters=("acetylcholine", "acetylcholine", "dopamine"),
        superclasses=("central",) * 3,
        outgoing=csr_array(
            (np.array([200.0], dtype=np.float32), ([0], [1])), shape=(3, 3)
        ),
    )


def calibration_binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([0], dtype=np.int64), canonical_edge_count=1
        ),
        pre_ids=np.array([10], dtype=np.uint64),
        post_ids=np.array([20], dtype=np.uint64),
    )


def test_paired_contact_changes_only_the_sparse_overlay_and_replays_exactly() -> None:
    config = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=1.0, refractory_ms=2.0, synaptic_delay_ms=1.0),
        learning_parameters=MushroomBodyLearningParameters(learning_rate=0.1),
    )
    result = run_associative_calibration(
        synthetic_calibration_graph(),
        calibration_binding(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        config=config,
    )

    assert result.classification == "plasticity_calibration"
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.final_multipliers[0] < 1.0
    assert result.contact_dan_events == 1


def test_unpaired_contact_has_no_learning_effect() -> None:
    config = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=1.0, refractory_ms=2.0, synaptic_delay_ms=1.0),
    )
    result = run_associative_calibration(
        synthetic_calibration_graph(),
        calibration_binding(),
        schedule=ConditioningSchedule.create(seed=7, steps=2),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        config=config,
    )

    assert result.final_multipliers == (1.0,)
    assert result.contact_dan_events == 0


def test_calibration_requires_an_explicit_dan_to_mbon_route() -> None:
    with pytest.raises(ValidationError):
        AssociativeCalibrationConfig(cue_ids=(10,), mbon_ids=(20,))


def test_opposite_dan_channel_cannot_modify_the_declared_appetitive_route() -> None:
    config = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=1.0, refractory_ms=2.0, synaptic_delay_ms=1.0),
    )
    result = run_associative_calibration(
        synthetic_calibration_graph(),
        calibration_binding(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, aversive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        config=config,
    )

    assert result.classification == "null"
    assert result.contact_dan_events == 1
    assert result.final_multipliers == (1.0,)
