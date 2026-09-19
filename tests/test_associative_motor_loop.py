import numpy as np
from scipy.sparse import csr_array

from flybrain.associative_motor_loop import (
    run_associative_motor_calibration,
    run_associative_motor_learning_probe,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.conditioning_world import ConditioningEvent, ConditioningSchedule
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS, HexapodMotorMap, MotorGroup
from flybrain.mushroom_body_learning import MushroomBodyLearningParameters
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.proprioceptive_interface import ProprioceptiveBank, ProprioceptiveMap
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters


def motor_map() -> HexapodMotorMap:
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


def proprio_map() -> ProprioceptiveMap:
    legs = ("left_fore", "right_fore", "left_middle", "right_middle", "left_hind", "right_hind")
    return ProprioceptiveMap(
        banks=tuple(
            ProprioceptiveBank(name=f"{leg}_proprioception", leg=leg, neuron_ids=(200 + index,))
            for index, leg in enumerate(legs)
        )
    )


def graph() -> EventConnectome:
    motor_ids = [100 + index for index in range(24)]
    proprio_ids = [200 + index for index in range(6)]
    ids = np.array([1, 10, 20, 30, *motor_ids, *proprio_ids], dtype=np.uint64)
    index = {int(value): position for position, value in enumerate(ids)}
    rows = [index[1], index[10], *([index[20]] * len(motor_ids))]
    cols = [index[10], index[20], *(index[item] for item in motor_ids)]
    values = np.array([400.0, 200.0, *([200.0] * len(motor_ids))], dtype=np.float32)
    return EventConnectome(
        neuron_ids=ids,
        cell_types=("olfactory", "Kenyon_Cell", "MBON", "DAN", *("motor",) * 24, *("proprio",) * 6),
        roles=(
            "learning_olfactory",
            "learning_kc",
            "learning_mbon",
            "dan_appetitive",
            *("motor",) * 24,
            *("sensory",) * 6,
        ),
        transmitters=("acetylcholine",) * len(ids),
        superclasses=("fixture",) * len(ids),
        outgoing=csr_array((values, (rows, cols)), shape=(len(ids), len(ids))),
    )


def binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1], dtype=np.int64),
            canonical_edge_count=26,
        ),
        pre_ids=np.array([10], dtype=np.uint64),
        post_ids=np.array([20], dtype=np.uint64),
    )


def test_associative_motor_loop_closes_neural_motor_body_proprioception_and_replays() -> None:
    result = run_associative_motor_calibration(
        graph(),
        binding(),
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,), mbon_ids=(20,), dan_to_mbon_pairs=((30, 20),),
            input_mode="sensory_path", sensory_input_ids=(1,), neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=1.0),
        ),
        motor=motor_map(), proprio=proprio_map(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.motor_spikes > 0
    assert result.mbon_spikes > 0
    assert 0 < result.sensory_voltage_events < 20
    assert result.proprioceptive_events == 12
    assert result.final_multipliers[0] < 1.0


def test_associative_motor_loop_accepts_shared_physics_backend() -> None:
    kwargs = dict(
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="direct_kc",
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    original = run_associative_motor_calibration(graph(), binding(), **kwargs)
    abstracted = run_associative_motor_calibration(
        graph(),
        binding(),
        backend_factory=ReferenceHexapodBackend,
        **kwargs,
    )

    assert abstracted == original


def test_motor_lesion_removes_only_motor_spikes_from_closed_loop() -> None:
    kwargs = dict(
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="direct_kc",
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        schedule=ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(0,)),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    normal = run_associative_motor_calibration(graph(), binding(), **kwargs)
    lesioned = run_associative_motor_calibration(
        graph(), binding(), motor_silenced=True, **kwargs
    )

    assert normal.motor_spikes > 0
    assert lesioned.motor_spikes == 0
    assert lesioned.proprioceptive_events == normal.proprioceptive_events
    assert lesioned.graph_unchanged is True


def test_no_contact_leaves_plastic_overlay_at_unity() -> None:
    result = run_associative_motor_calibration(
        graph(),
        binding(),
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="sensory_path",
            sensory_input_ids=(1,),
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        schedule=ConditioningSchedule(
            seed=7,
            events=(
                ConditioningEvent(
                    step=0,
                    odor_intensity=1.0,
                    appetitive_contact_intensity=0.0,
                    aversive_contact_intensity=0.0,
                ),
                ConditioningEvent(
                    step=1,
                    odor_intensity=0.0,
                    appetitive_contact_intensity=0.0,
                    aversive_contact_intensity=0.0,
                ),
            ),
        ),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.final_multipliers == (1.0,)


def test_proprioceptive_silence_removes_only_body_feedback_events() -> None:
    normal = run_associative_motor_calibration(
        graph(),
        binding(),
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="sensory_path",
            sensory_input_ids=(1,),
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        schedule=ConditioningSchedule.create(
            seed=7,
            steps=2,
            appetitive_pair_steps=(0,),
        ),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )
    silenced = run_associative_motor_calibration(
        graph(),
        binding(),
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="sensory_path",
            sensory_input_ids=(1,),
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        schedule=ConditioningSchedule.create(
            seed=7,
            steps=2,
            appetitive_pair_steps=(0,),
        ),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
        proprioceptive_silenced=True,
    )

    assert normal.proprioceptive_events == 12
    assert silenced.proprioceptive_events == 0
    assert silenced.graph_unchanged is True


def test_learning_probe_isolates_persistent_sparse_weight_effect() -> None:
    result = run_associative_motor_learning_probe(
        graph(),
        binding(),
        learning=AssociativeCalibrationConfig(
            cue_ids=(10,),
            mbon_ids=(20,),
            dan_to_mbon_pairs=((30, 20),),
            input_mode="direct_kc",
            neural_chunk_steps=20,
            shiu_parameters=ShiuParameters(
                dt_ms=0.5,
                refractory_ms=2.0,
                synaptic_delay_ms=1.0,
            ),
            learning_parameters=MushroomBodyLearningParameters(
                learning_rate=0.8,
                minimum_multiplier=0.2,
            ),
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        training_schedule=ConditioningSchedule.create(
            seed=7,
            steps=2,
            appetitive_pair_steps=(0,),
        ),
        probe_schedule=ConditioningSchedule(
            seed=7,
            events=(
                ConditioningEvent(
                    step=0,
                    odor_intensity=1.0,
                    appetitive_contact_intensity=0.0,
                    aversive_contact_intensity=0.0,
                ),
                ConditioningEvent(
                    step=1,
                    odor_intensity=0.0,
                    appetitive_contact_intensity=0.0,
                    aversive_contact_intensity=0.0,
                ),
            ),
        ),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.autonomous_behavior_claim_allowed is False
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert abs(result.training_final_multipliers[0] - 0.2) < 1e-6
    assert result.baseline_probe_motor_spikes > result.learned_probe_motor_spikes
    assert result.classification == "motor_difference"
