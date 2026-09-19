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
from flybrain.olfactory_interface import OlfactoryReceptorMap
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_edge_registry import ResolvedPlasticEdgeManifest
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


def sensory_calibration_graph() -> EventConnectome:
    return EventConnectome(
        neuron_ids=np.array([1, 10, 20, 30], dtype=np.uint64),
        cell_types=("ORN_DA1", "Kenyon_Cell", "MBON", "DAN"),
        roles=("learning_olfactory", "learning_kc", "learning_mbon", "dan_appetitive"),
        transmitters=("acetylcholine", "acetylcholine", "acetylcholine", "dopamine"),
        superclasses=("central",) * 4,
        outgoing=csr_array(
            (
                np.array([200.0, 200.0], dtype=np.float32),
                ([0, 1], [1, 2]),
            ),
            shape=(4, 4),
        ),
    )


def sensory_calibration_binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1], dtype=np.int64), canonical_edge_count=2
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
    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.final_multipliers[0] < 1.0
    assert result.contact_dan_events == 1


def test_transmitter_gain_profile_is_explicit_and_validated() -> None:
    config = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        presynaptic_transmitter_multipliers={"acetylcholine": 0.5, "gaba": 1.2},
    )

    assert config.presynaptic_transmitter_multipliers == {
        "acetylcholine": 0.5,
        "gaba": 1.2,
    }
    with pytest.raises(ValidationError, match="transmitter multipliers"):
        AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            presynaptic_transmitter_multipliers={"acetylcholine": -1.0},
        )


def test_manifest_config_resolves_a_declared_orn_channel_pattern() -> None:
    receptor_map = OlfactoryReceptorMap.from_graph(
        sensory_calibration_graph(), olfactory_neuron_ids=(1,)
    )
    dan_manifest = ResolvedPlasticEdgeManifest(
        name="dan_to_mbon",
        sign=0,
        edge_count=1,
        contact_count=1,
        pre_count=1,
        post_count=1,
        edge_sha256="a" * 64,
        edge_pairs=((30, 20),),
    )
    config = AssociativeCalibrationConfig.from_manifest(
        sensory_calibration_binding(),
        dan_manifest,
        ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        valence="appetitive",
        olfactory_receptor_map=receptor_map,
        olfactory_channel_types=("ORN_DA1",),
    )

    assert config.input_mode == "sensory_path"
    assert config.sensory_input_ids == (1,)
    assert config.sensory_channel_types == ("ORN_DA1",)


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


def test_sensory_path_calibration_avoids_direct_external_kc_drive() -> None:
    config = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        input_mode="sensory_path",
        sensory_input_ids=(1,),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=1.0, refractory_ms=2.0, synaptic_delay_ms=1.0),
    )
    result = run_associative_calibration(
        sensory_calibration_graph(),
        sensory_calibration_binding(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        config=config,
    )

    assert result.input_mode == "sensory_path"
    assert result.cue_spikes > 0
    assert result.mbon_spikes > 0
    assert result.final_multipliers[0] < 1.0


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


def test_config_can_bind_appetitive_routes_only_from_a_zero_sign_manifest() -> None:
    manifest = ResolvedPlasticEdgeManifest(
        name="dan_to_mbon",
        sign=0,
        edge_count=2,
        contact_count=2,
        pre_count=2,
        post_count=1,
        edge_sha256="a" * 64,
        edge_pairs=((30, 20), (31, 20)),
    )

    config = AssociativeCalibrationConfig.from_manifest(
        calibration_binding(),
        manifest,
        ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        valence="appetitive",
    )

    assert config.cue_ids == (10,)
    assert config.mbon_ids == (20,)
    assert config.dan_to_mbon_pairs == ((30, 20),)
