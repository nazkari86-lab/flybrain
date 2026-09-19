import pytest
from test_associative_motor_loop import binding, graph, motor_map, proprio_map

from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    HexapodArenaConfig,
    run_autonomous_hexapod_episode,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_perturbations import BodyPerturbation
from flybrain.flygym_backend import FlyGymBackend, flygym_availability
from flybrain.hexapod_body import HexapodParameters
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters


def config() -> AutonomousHexapodConfig:
    return AutonomousHexapodConfig(
        body_steps=2,
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
        arena=HexapodArenaConfig(
            food_position_m=(0.0, 0.0),
            threat_position_m=(10.0, 10.0),
            contact_radius_m=0.05,
            odor_length_scale_m=1.0,
        ),
        seed=7,
    )


def test_contact_driven_hexapod_closes_learning_motor_body_feedback_loop() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.replay_exact is True
    assert result.backend.name == "reference_hexapod"
    assert result.backend.evidence_kind == "model_assumption"
    assert result.graph_unchanged is True
    assert result.appetitive_contacts == 2
    assert result.aversive_contacts == 0
    assert result.dan_events == 2
    assert result.motor_spikes > 0
    assert result.active_motor_groups == 24
    assert result.all_motor_groups_active is True
    assert result.proprioceptive_events == 12
    assert result.final_multipliers[0] < 1.0
    assert result.autonomous_behavior_claim_allowed is False


def test_contact_driven_episode_integrates_local_slow_dan_no_memory() -> None:
    baseline = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )
    slow_config = config().model_copy(update={"nitric_oxide_dan_ids": (30,)})

    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        slow_config,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.replay_exact is True
    assert result.slow_memory_enabled is True
    assert result.slow_memory_edges == 1
    assert result.slow_memory_dopamine_effect_max > 0.0
    assert result.slow_memory_nitric_oxide_effect_max > 0.0
    assert result.final_multipliers != baseline.final_multipliers


def test_autonomous_hexapod_contract_has_no_schedule_reward_or_target_output() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    forbidden = {"schedule", "reward", "target", "desired_action", "policy"}
    assert forbidden.isdisjoint(type(result).model_fields)


def test_autonomous_episode_reports_environment_trace_under_perturbation() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
        perturbation=BodyPerturbation(delay_steps=1, damaged_legs=(0,)),
    )

    assert len(result.trace_distance_to_food) == 2
    assert len(result.trace_distance_to_threat) == 2
    assert result.first_food_contact_step == 0
    assert result.first_threat_contact_step is None


@pytest.mark.skipif(
    not flygym_availability().available,
    reason=flygym_availability().reason,
)
def test_same_autonomous_neural_loop_runs_on_flygym_without_retraining() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
        backend_factory=FlyGymBackend,
    )

    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert result.all_motor_groups_active is True
    assert result.proprioceptive_events == 12
    assert result.final_multipliers[0] < 1.0
