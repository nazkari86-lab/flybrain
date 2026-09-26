from test_associative_motor_loop import binding, graph, motor_map, proprio_map

from flybrain.autonomous_behavior_benchmark import (
    BehaviorBenchmarkConfig,
    _episode_seed,
    run_behavior_benchmark,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_perturbations import BehaviorVariant
from flybrain.descending_interface import DescendingMap
from flybrain.hexapod_body import HexapodParameters
from flybrain.olfactory_interface import TaskOdorAssignment
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
    second_variant = BehaviorVariant(
        name="train-threat",
        food_position_m=(10.0, 10.0),
        threat_position_m=(0.0, 0.0),
    )
    config = BehaviorBenchmarkConfig(
        training_episodes=1,
        holdout_episodes=1,
        body_steps=2,
        seeds=(7, 8),
        training_variants=(variant, second_variant),
        holdout_variants=(
            variant.model_copy(update={"name": "holdout-food"}),
            second_variant.model_copy(update={"name": "holdout-threat"}),
        ),
        descending_map=DescendingMap(
            d_na02_left=(10,),
            d_na02_right=(20,),
            d_ng13_left=(100,),
            d_ng13_right=(101,),
            mdn_left=(102,),
            mdn_right=(103,),
        ),
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
    assert result.protocol == "autonomous-behavior-benchmark-v2"
    assert result.evidence_protocol == "measured-replay-persistent-memory-v6"
    assert result.graph_unchanged is True
    assert result.replay_exact is True
    assert result.holdout_weights_frozen is True
    assert result.generalization_verified is False
    assert len(result.world_split.overlap_digests) == 2
    assert result.world_split.holdout_target_world_counts == {"food": 2, "threat": 2}
    assert result.unassisted_motor_output is True
    assert result.behavioral_horizon_adequate is False
    assert set(result.causal_diagnostics) == {
        "no_plasticity",
        "dan_lesion",
        "kc_mbon_lesion",
        "rewired_control",
    }
    assert all(
        item.paired_holdout_episodes == 4
        for item in result.causal_diagnostics.values()
    )
    assert len(result.rewired_control_evidence) == 2
    assert all(item.changed_edges == 0 for item in result.rewired_control_evidence)
    assert result.rewired_control_effective is False
    assert len(result.episode_evidence) == 40
    assert all(item.replay_performed and item.replay_exact for item in result.episode_evidence)
    assert all(
        item.weights_unchanged and not item.plasticity_enabled
        for item in result.episode_evidence if item.holdout
    )
    assert result.behavioral_claim_allowed is False
    assert result.odor_channel_model == "bipartite_registered_olfactory_assumption"
    assert all(len(items) == 4 for items in result.condition_observations.values())
    assert result.training_summaries["normal"].appetitive_contacts > 0
    assert result.training_summaries["normal"].aversive_contacts > 0
    assert result.training_summaries["normal"].dan_events > 0
    assert result.training_summaries["dan_lesion"].dan_events == 0
    assert len(result.holdout_neural_activity["normal"]) == 4
    assert len(result.training_neural_activity["normal"]) == 4
    assert all(
        item.mbon_spikes == sum(item.mbon_spike_counts.values())
        for item in result.training_neural_activity["normal"]
    )
    assert all(
        sum(item.dan_spike_counts.values()) == 0
        for item in result.training_neural_activity["dan_lesion"]
    )
    normal_activity = result.holdout_neural_activity["normal"]
    assert sum(item.plastic_kc_spikes for item in normal_activity) > 0
    assert sum(item.mbon_spikes for item in normal_activity) > 0
    assert all(item.motor_spikes >= 0 for item in normal_activity)


def test_benchmark_propagates_subtype_encoder_through_training_and_holdout() -> None:
    learning = AssociativeCalibrationConfig(
        cue_ids=(10,), mbon_ids=(20,), dan_to_mbon_pairs=((30, 20),),
        input_mode="sensory_path", sensory_input_ids=(1,), neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=1.0),
    )
    food = BehaviorVariant(
        name="food-training", food_position_m=(0.0, 0.0), threat_position_m=(10.0, 10.0)
    )
    threat = BehaviorVariant(
        name="threat-training", food_position_m=(10.0, 10.0), threat_position_m=(0.0, 0.0)
    )
    configuration = BehaviorBenchmarkConfig(
        training_episodes=1, holdout_episodes=1, body_steps=2, seeds=(7,),
        controls=("no_plasticity",),
        training_variants=(food, threat),
        holdout_variants=(
            food.model_copy(update={"name": "food-holdout", "evaluation_target": "food"}),
            threat.model_copy(update={"name": "threat-holdout", "evaluation_target": "threat"}),
        ),
        proprioceptive_encoding="subtype_weighted_spikes",
    )
    labels = {neuron_id: "leg" for bank in proprio_map().banks for neuron_id in bank.neuron_ids}

    result = run_behavior_benchmark(
        graph(), binding(), configuration,
        learning=learning, motor=motor_map(), proprio=proprio_map(),
        reinforcement=ReinforcementInterface(appetitive_dan_ids=(30,), aversive_dan_ids=(31,)),
        body_parameters=HexapodParameters(dt_s=0.01),
        proprioceptive_subtypes=labels,
    )

    assert result.proprioceptive_encoding == "subtype_weighted_spikes"
    assert result.replay_exact
    assert len(result.episode_evidence) == 8
    assert all(item.replay_exact for item in result.episode_evidence)
    assert result.behavioral_claim_allowed is False


def test_generalization_requires_two_unseen_worlds_per_task() -> None:
    learning = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        input_mode="sensory_path",
        sensory_input_ids=(1,),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(
            dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=1.0
        ),
    )
    training = BehaviorVariant(
        name="training",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    holdouts = (
        BehaviorVariant(
            name="food-a", evaluation_target="food",
            food_position_m=(0.2, 0.1), threat_position_m=(9.0, 9.0),
        ),
        BehaviorVariant(
            name="food-b", evaluation_target="food",
            food_position_m=(-0.2, 0.1), threat_position_m=(9.0, -9.0),
        ),
        BehaviorVariant(
            name="threat-a", evaluation_target="threat",
            food_position_m=(9.0, 9.0), threat_position_m=(0.1, 0.0),
        ),
        BehaviorVariant(
            name="threat-b", evaluation_target="threat",
            food_position_m=(-9.0, 9.0), threat_position_m=(-0.1, 0.0),
        ),
    )
    result = run_behavior_benchmark(
        graph(),
        binding(),
        BehaviorBenchmarkConfig(
            training_episodes=1,
            holdout_episodes=1,
            body_steps=1,
            seeds=(7,),
            training_variants=(training,),
            holdout_variants=holdouts,
        ),
        learning=learning,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.generalization_verified is True
    assert result.world_split.overlap_digests == ()
    assert result.world_split.holdout_target_world_counts == {"food": 2, "threat": 2}


def test_behavioral_claim_gate_rejects_contact_recruited_learning(
    monkeypatch,
) -> None:
    import flybrain.autonomous_behavior_benchmark as benchmark

    monkeypatch.setattr(benchmark, "claim_gate", lambda *args, **kwargs: True)
    learning = AssociativeCalibrationConfig(
        cue_ids=(10,),
        mbon_ids=(20,),
        dan_to_mbon_pairs=((30, 20),),
        input_mode="sensory_path",
        sensory_input_ids=(1,),
        neural_chunk_steps=20,
        shiu_parameters=ShiuParameters(
            dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=1.0
        ),
    )
    training_food = BehaviorVariant(
        name="training-food",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    training_threat = BehaviorVariant(
        name="training-threat",
        food_position_m=(10.0, 10.0),
        threat_position_m=(0.0, 0.0),
    )
    config = BehaviorBenchmarkConfig(
        training_episodes=1,
        holdout_episodes=1,
        body_steps=1,
        seeds=(7,),
        training_variants=(training_food, training_threat),
        holdout_variants=(
            training_food.model_copy(update={
                "name": "holdout-food",
                "evaluation_target": "food",
                "food_position_m": (0.2, 0.1),
            }),
            training_threat.model_copy(update={
                "name": "holdout-threat",
                "evaluation_target": "threat",
                "threat_position_m": (0.1, 0.0),
            }),
        ),
        odor_assignment=TaskOdorAssignment(
            food_cell_types=("ORN_DM1",),
            threat_cell_types=("ORN_DA2",),
            food_evidence_doi="food",
            threat_evidence_doi="threat",
        ),
    )
    result = benchmark.run_behavior_benchmark(
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

    assert config.reinforcement_source == "contact_recruited"
    assert result.behavioral_claim_allowed is False


def test_generalization_rejects_training_world_reused_as_holdout() -> None:
    variant = BehaviorVariant(
        name="same-world",
        food_position_m=(0.0, 0.0),
        threat_position_m=(10.0, 10.0),
    )
    config = BehaviorBenchmarkConfig(
        training_episodes=1,
        holdout_episodes=1,
        body_steps=1,
        seeds=(7,),
        training_variants=(variant,),
        holdout_variants=(
            variant.model_copy(update={"name": "renamed", "evaluation_target": "food"}),
            variant.model_copy(update={"name": "renamed-2", "evaluation_target": "threat"}),
        ),
    )

    assert config.world_split().unseen_worlds is False
    assert config.world_split().overlap_digests


def test_episode_seed_does_not_alias_internal_sensory_stream_offsets() -> None:
    first = _episode_seed(7, 0)
    second = _episode_seed(7, 1)
    stream_offsets = (0, 1_000_003, 2_000_006, 3_000_009, 3_000_017, 4_000_019)

    assert second not in {first + offset for offset in stream_offsets}
    assert _episode_seed(7, 1) == second
    assert _episode_seed(8, 1) != second
