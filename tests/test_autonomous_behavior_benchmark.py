from test_associative_motor_loop import binding, graph, motor_map, proprio_map

from flybrain.autonomous_behavior_benchmark import (
    BehaviorBenchmarkConfig,
    run_behavior_benchmark,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_perturbations import BehaviorVariant
from flybrain.hexapod_body import HexapodParameters
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters


def test_benchmark_persists_isolated_conditions_and_preserves_graph() -> None:
    learning = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        input_mode="sensory_path",
        sensory_input_ids=(1,),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=1.0),
    )
    variant = BehaviorVariant(
        name="train-a",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    config = BehaviorBenchmarkConfig(
        training_episodes=1,
        holdout_episodes=1,
        body_steps=2,
        seeds=(7, 8),
        training_variants=(variant,),
        holdout_variants=(variant.model_copy(update={"name": "holdout-b"}),),
    )
    result = run_behavior_benchmark(
        graph(),
        binding(),
        config,
        learning=learning,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert set(result.condition_observations) == {
        "normal",
        "no_plasticity",
        "dan_lesion",
        "kc_mbon_lesion",
        "rewired_control",
    }
    assert result.graph_unchanged is True
    assert result.behavioral_claim_allowed is False
    assert all(len(items) == 1 for items in result.condition_observations.values())
