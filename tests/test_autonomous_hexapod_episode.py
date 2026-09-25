import numpy as np
import pytest
from scipy.sparse import csr_array
from test_associative_motor_loop import binding, graph, motor_map, proprio_map

from flybrain.autonomous_hexapod_episode import (
    AutonomousHexapodConfig,
    HexapodArenaConfig,
    _anonymous_visual_scene,
    _bilateral_odor_intensities,
    _odor_intensities,
    run_autonomous_hexapod_episode,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.behavioral_perturbations import BodyPerturbation
from flybrain.descending_interface import DescendingMap
from flybrain.flygym_backend import FlyGymBackend, flygym_availability
from flybrain.graph import EventConnectome
from flybrain.hexapod_backend import ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.retinal_interface import VisualInterfaceMap
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


def test_temporal_plume_changes_concentration_without_privileged_action() -> None:
    arena = config().arena
    body = ReferenceHexapodBackend(HexapodParameters()).observe()
    first = _odor_intensities(body, arena, 0.0)
    later = _odor_intensities(body, arena, 0.0625)
    assert first != later
    assert all(0.0 <= value <= 1.0 for value in (*first, *later))


def test_bilateral_food_and_threat_channels_preserve_declared_plume_phase() -> None:
    arena = HexapodArenaConfig(
        food_position_m=(1.0, 0.0),
        threat_position_m=(1.0, 0.0),
        contact_radius_m=0.05,
        odor_length_scale_m=1.0,
    )
    body = ReferenceHexapodBackend(HexapodParameters()).observe()

    food_left, food_right, threat_left, threat_right = _bilateral_odor_intensities(
        body, arena, 0.0
    )

    assert threat_left > food_left
    assert threat_right > food_right


def test_arena_uses_declared_antenna_spacing_and_visual_disc_size() -> None:
    body = ReferenceHexapodBackend(HexapodParameters()).observe()
    arena = HexapodArenaConfig(
        food_position_m=(1.0, 0.2),
        threat_position_m=(2.0, -0.3),
        contact_radius_m=0.01,
        odor_length_scale_m=1.0,
        antenna_lateral_offset_m=0.0,
        visual_disc_radius_m=0.001,
    )

    food_left, food_right, threat_left, threat_right = _bilateral_odor_intensities(
        body, arena
    )
    assert food_left == pytest.approx(food_right)
    assert threat_left == pytest.approx(threat_right)
    assert all(disc.radius == pytest.approx(0.001) for disc in _anonymous_visual_scene(body, arena))


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
    assert result.active_motor_groups == 36
    assert result.all_motor_groups_active is True
    assert result.proprioceptive_events == 12
    assert result.final_multipliers[0] < 1.0
    assert result.autonomous_behavior_claim_allowed is False


def test_neural_dan_reinforcement_does_not_learn_from_contact_without_dan_spikes() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config().model_copy(update={
            "reinforcement_source": "contact_gated_neural_dan",
        }),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.appetitive_contacts == 2
    assert result.dan_events == 2
    assert result.routed_dan_spike_events == 0
    assert result.final_multipliers == (1.0,)


def test_autonomous_result_records_no_external_phase_drive() -> None:
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

    assert result.phase_envelope.amplitude_nm == 0.0
    assert result.phase_envelope.evidence_kind == "model_assumption"
    assert result.neural_authority_nm[1] < result.neural_authority_nm[0]
    assert result.neural_authority_nm[2] < result.neural_authority_nm[0]


def test_episode_reports_learning_path_spikes_before_motor_output() -> None:
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

    assert result.plastic_kc_spikes > 0
    assert result.mbon_spikes > 0
    assert result.motor_spikes > 0
    assert result.mbon_spike_counts == {20: 1}
    assert result.mbon_mean_multipliers[20] == pytest.approx(0.7604638934)


def test_episode_can_use_source_equivalent_proprioceptive_spikes() -> None:
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config().model_copy(
            update={
                "proprioceptive_encoding": "source_equivalent_spikes",
                "proprioceptive_spike_rate_hz": 30.0,
            }
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.proprioceptive_events > 0
    assert result.proprioceptive_events != 12
    assert result.proprioceptive_encoding == "source_equivalent_spikes"
    assert result.proprioceptive_spike_rate_hz == pytest.approx(30.0)
    assert result.replay_exact is True
    assert result.graph_unchanged is True


def test_episode_reports_registered_descending_spikes_without_decoding_commands() -> None:
    descending = DescendingMap(
        d_na02_left=(10,),
        d_na02_right=(20,),
        d_ng13_left=(100,),
        d_ng13_right=(101,),
        mdn_left=(102,),
        mdn_right=(103,),
    )
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        config().model_copy(update={"descending_map": descending}),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.descending_spikes.d_na02_left > 0
    assert result.descending_spikes.d_na02_right > 0
    assert result.descending_spikes.d_ng13_left > 0
    assert result.descending_spikes.d_ng13_right > 0
    assert result.descending_spikes.mdn_left > 0
    assert result.descending_spikes.mdn_right > 0


def test_explicit_olfactory_channels_control_which_declared_source_drives_the_loop() -> None:
    base = graph()
    coo = base.outgoing.tocoo()
    rows = np.where(coo.row >= 1, coo.row + 1, coo.row)
    cols = np.where(coo.col >= 1, coo.col + 1, coo.col)
    dual = EventConnectome(
        neuron_ids=np.insert(base.neuron_ids, 1, np.uint64(2)),
        cell_types=(*base.cell_types[:1], "olfactory", *base.cell_types[1:]),
        roles=(*base.roles[:1], "learning_olfactory", *base.roles[1:]),
        transmitters=(*base.transmitters[:1], "acetylcholine", *base.transmitters[1:]),
        superclasses=(*base.superclasses[:1], "fixture", *base.superclasses[1:]),
        outgoing=csr_array(
            (coo.data, (rows, cols)),
            shape=(base.neuron_count + 1, base.neuron_count + 1),
        ),
    )
    learning = config().learning.model_copy(update={"sensory_input_ids": (1, 2)})
    explicit = config().model_copy(
        update={
            "learning": learning,
            "odor_a_input_ids": (1,),
            "odor_b_input_ids": (2,),
        }
    )
    swapped = explicit.model_copy(
        update={"odor_a_input_ids": (2,), "odor_b_input_ids": (1,)}
    )
    kwargs = dict(
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    driven = run_autonomous_hexapod_episode(dual, binding(), explicit, **kwargs)
    silent = run_autonomous_hexapod_episode(dual, binding(), swapped, **kwargs)

    assert driven.odor_channel_model == "receptor_bank_olfactory_assumption"
    assert driven.motor_spikes > silent.motor_spikes


def test_bilateral_olfactory_channels_reject_mixed_primary_and_invalid_ids() -> None:
    base = config().model_dump()
    base["learning"]["sensory_input_ids"] = (1, 2, 3, 4)
    bilateral = {
        "odor_a_left_input_ids": (1,),
        "odor_a_right_input_ids": (2,),
        "odor_b_left_input_ids": (3,),
        "odor_b_right_input_ids": (4,),
    }

    with pytest.raises(ValueError, match="cannot mix"):
        AutonomousHexapodConfig(**{
            **base,
            "odor_a_input_ids": (1, 2),
            "odor_b_input_ids": (3, 4),
            **bilateral,
        })
    with pytest.raises(ValueError, match="unique and sorted"):
        AutonomousHexapodConfig(**{
            **base, **bilateral, "odor_a_left_input_ids": (2, 1)
        })
    with pytest.raises(ValueError, match="disjoint"):
        AutonomousHexapodConfig(**{
            **base, **bilateral, "odor_b_left_input_ids": (1,)
        })


def test_bilateral_ids_must_be_declared_sensory_before_graph_lookup() -> None:
    base = config().model_dump()
    base["learning"]["sensory_input_ids"] = (1, 2, 3, 4)
    with pytest.raises(ValueError, match="declared sensory"):
        AutonomousHexapodConfig(**{
            **base,
            "odor_a_left_input_ids": (1,),
            "odor_a_right_input_ids": (2,),
            "odor_b_left_input_ids": (3,),
            "odor_b_right_input_ids": (999,),
        })


def test_disabled_plasticity_keeps_the_live_overlay_at_its_initial_values() -> None:
    live_binding = binding()

    result = run_autonomous_hexapod_episode(
        graph(),
        live_binding,
        config(),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
        mutate_binding=True,
        plasticity_enabled=False,
    )

    assert result.dan_events == 2
    assert result.final_multipliers == (1.0,)
    assert tuple(live_binding.overlay.multipliers) == (1.0,)


def test_autonomous_episode_can_feed_anonymous_retinal_source_spikes() -> None:
    visual_config = config().model_copy(
        update={
            "body_steps": 20,
            "arena": HexapodArenaConfig(
                food_position_m=(0.06, 0.001),
                threat_position_m=(10.0, 10.0),
                contact_radius_m=0.05,
                odor_length_scale_m=1.0,
            ),
            "visual_interface": VisualInterfaceMap(
                left_r1_r6_ids=(1,),
                right_r1_r6_ids=(10,),
                left_hs_ids=(20,),
                right_hs_ids=(30,),
                left_lc16_ids=(100,),
                right_lc16_ids=(200,),
            ),
        }
    )

    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        visual_config,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.visual_source_events > 0
    assert result.replay_exact is True
    assert result.graph_unchanged is True


def test_retinal_source_neuron_ids_are_translated_to_csr_indices() -> None:
    visual_config = config().model_copy(
        update={
            "body_steps": 20,
            "arena": HexapodArenaConfig(
                food_position_m=(0.06, 0.001),
                threat_position_m=(10.0, 10.0),
                contact_radius_m=0.05,
                odor_length_scale_m=1.0,
            ),
            "visual_interface": VisualInterfaceMap(
                left_r1_r6_ids=(200,),
                right_r1_r6_ids=(205,),
                left_hs_ids=(1,),
                right_hs_ids=(10,),
                left_lc16_ids=(20,),
                right_lc16_ids=(30,),
            ),
        }
    )

    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        visual_config,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.visual_source_events > 0
    assert result.replay_exact is True


def test_physical_contact_gates_registered_tactile_inputs() -> None:
    tactile_config = config().model_copy(update={"tactile_contact_input_ids": (1,)})
    result = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        tactile_config,
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert result.tactile_contact_events > 0
    assert result.autonomous_behavior_claim_allowed is False

    no_contact = run_autonomous_hexapod_episode(
        graph(),
        binding(),
        tactile_config.model_copy(
            update={
                "arena": tactile_config.arena.model_copy(
                    update={"food_position_m": (10.0, 10.0)}
                )
            }
        ),
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )

    assert no_contact.tactile_contact_events == 0


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


def test_motor_diagnostic_records_applied_torque_after_delay_and_damage() -> None:
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
        capture_motor_trace=True,
    )

    assert result.replay_exact is True
    assert result.motor_trace is not None
    assert len(result.motor_trace) == 2
    first, second = result.motor_trace
    assert len(first.activations) == 36
    assert first.applied_torque_nm.values == ((0.0, 0.0, 0.0),) * 6
    assert second.applied_torque_nm.values[0] == (0.0, 0.0, 0.0)
    assert second.applied_torque_nm.values[1:] == first.decoded_torque_nm.values[1:]
    assert second.thorax_position_m == result.final_body.thorax_position_m
    assert second.thorax_yaw_rad == result.final_body.thorax_yaw_rad
    assert second.support_count == result.final_body.support_count


def test_motor_diagnostic_is_off_by_default() -> None:
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

    assert result.motor_trace is None


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
