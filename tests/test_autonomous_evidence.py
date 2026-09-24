"""Regression checks for evidence that must be measured, never assumed."""

import numpy as np
import pytest
from test_associative_motor_loop import binding, graph, motor_map, proprio_map
from test_autonomous_hexapod_episode import config

from flybrain.autonomous_behavior_benchmark import (
    BehaviorBenchmarkConfig,
    run_behavior_benchmark,
)
from flybrain.autonomous_hexapod_episode import run_autonomous_hexapod_episode
from flybrain.behavioral_controls import build_condition
from flybrain.behavioral_perturbations import BehaviorVariant
from flybrain.embodied_interfaces import ExternalEvent
from flybrain.hexapod_backend import ReferenceHexapodBackend
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorDecoder
from flybrain.olfactory_interface import TaskOdorAssignment
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.retinal_interface import VisualInterfaceMap


def episode_kwargs():
    return dict(
        motor=motor_map(),
        proprio=proprio_map(),
        reinforcement=ReinforcementInterface(
            appetitive_dan_ids=(30,), aversive_dan_ids=(31,)
        ),
        body_parameters=HexapodParameters(dt_s=0.01),
    )


def benchmark_config(**updates):
    # Contact during evaluation deliberately exposes accidental test-time learning.
    variant = BehaviorVariant(
        name="contact", food_position_m=(0.0, 0.0), threat_position_m=(10.0, 10.0)
    )
    return BehaviorBenchmarkConfig(**dict(
        training_episodes=1, holdout_episodes=1, body_steps=2,
        seeds=(7,), training_variants=(variant,), holdout_variants=(variant,),
        **updates,
    ))


def test_skipped_replay_cannot_count_as_exact():
    result = run_autonomous_hexapod_episode(
        graph(), binding(), config(), replay=False, **episode_kwargs()
    )
    assert result.replay_exact is False


def test_learning_replay_uses_pre_training_weights_without_double_learning():
    original = binding()
    first = run_autonomous_hexapod_episode(
        graph(), original, config(), replay=False, mutate_binding=True, **episode_kwargs()
    )
    replayed_binding = binding()
    replayed = run_autonomous_hexapod_episode(
        graph(), replayed_binding, config(), replay=True, mutate_binding=True,
        **episode_kwargs(),
    )
    assert replayed.replay_exact is True
    assert replayed.final_multipliers == first.final_multipliers
    np.testing.assert_array_equal(
        original.overlay.multipliers, replayed_binding.overlay.multipliers
    )


def test_benchmark_detects_a_backend_that_changes_between_repeats():
    creations = 0

    def drifting_backend(parameters):
        nonlocal creations
        creations += 1
        return ReferenceHexapodBackend(parameters.model_copy(update={
            "initial_thorax_position_m": (creations * 0.001, 0.0, 0.002),
        }))

    result = run_behavior_benchmark(
        graph(), binding(), benchmark_config(), learning=config().learning,
        backend_factory=drifting_backend, **episode_kwargs(),
    )
    assert result.replay_exact is False
    assert result.behavioral_claim_allowed is False


def test_holdout_contact_does_not_change_learned_weights(monkeypatch):
    from flybrain import autonomous_behavior_benchmark as benchmark

    evaluations = []
    run_episode = benchmark.run_autonomous_hexapod_episode

    def observe_episode(graph, state, config, **kwargs):
        before = tuple(float(x) for x in state.overlay.multipliers)
        result = run_episode(graph, state, config, **kwargs)
        if not kwargs["mutate_binding"]:
            evaluations.append((before, result.final_multipliers))
        return result

    monkeypatch.setattr(benchmark, "run_autonomous_hexapod_episode", observe_episode)
    run_behavior_benchmark(
        graph(), binding(), benchmark_config(), learning=config().learning,
        **episode_kwargs(),
    )
    assert evaluations
    assert all(before == after for before, after in evaluations)


def test_behavioral_claim_rejects_physically_fallen_holdouts(monkeypatch):
    from flybrain import autonomous_behavior_benchmark as benchmark

    original = benchmark.run_autonomous_hexapod_episode

    def fallen_episode(*args, **kwargs):
        result = original(*args, **kwargs)
        body = result.final_body.model_copy(update={"fallen": True, "support_count": 0})
        return result.model_copy(update={"final_body": body})

    monkeypatch.setattr(benchmark, "run_autonomous_hexapod_episode", fallen_episode)
    monkeypatch.setattr(benchmark, "claim_gate", lambda *args, **kwargs: True)
    settings = benchmark_config().model_copy(update={
        "reinforcement_source": "contact_gated_neural_dan",
        "odor_assignment": TaskOdorAssignment(
            food_cell_types=("ORN_DM1",),
            threat_cell_types=("ORN_DA2",),
            food_evidence_doi="food",
            threat_evidence_doi="threat",
        ),
    })

    result = benchmark.run_behavior_benchmark(
        graph(), binding(), settings, learning=config().learning, **episode_kwargs()
    )

    assert result.holdout_physical_stability_verified is False
    assert result.behavioral_claim_allowed is False


def test_duplicate_seeds_do_not_count_as_independent_replicates():
    values = benchmark_config().model_dump()
    values["seeds"] = (7, 7, 7)
    with pytest.raises(ValueError, match="unique"):
        BehaviorBenchmarkConfig(**values)


def test_distance_baseline_is_measured_before_first_motor_step():
    from flybrain.autonomous_behavior_benchmark import _observation

    class MovingBackend(ReferenceHexapodBackend):
        shifted = None

        def observe(self):
            return self.shifted if self.shifted is not None else super().observe()

        def step(self, torque):
            self.shifted = super().step(torque).model_copy(update={
                "thorax_position_m": (0.01, 0.0, 0.002),
            })
            return self.shifted

    settings = config().model_copy(update={"body_steps": 1})
    result = run_autonomous_hexapod_episode(
        graph(), binding(), settings, backend_factory=MovingBackend, **episode_kwargs(),
    )
    observation = _observation(result, horizon=1, evaluation_target="both")
    assert observation.initial_food_distance == 0.0
    assert observation.initial_threat_distance == pytest.approx(200.0 ** 0.5)
    assert observation.final_food_distance == 0.01


def test_kc_mbon_lesion_does_not_also_lesion_dopamine_neurons():
    control = build_condition("kc_mbon_lesion", binding(), config().learning, seed=7)
    assert control.dan_enabled is True
    assert not np.any(control.overlay.multipliers)


def test_default_autonomous_decoder_cannot_drive_muscles_without_spikes():
    for parameters in (config(), benchmark_config()):
        decoder = HexapodMotorDecoder(
            motor_map(), phase_envelope=parameters.phase_envelope,
            neural_authority_nm=parameters.neural_authority_nm,
        )
        state = decoder.decode((), window_s=0.01, leg_phases=(0.25,) * 6)
        assert state.torques.values == ((0.0, 0.0, 0.0),) * 6


def test_coincident_odor_input_preserves_retinal_spikes_and_dan_lesion_suppresses_them(
    monkeypatch,
):
    from flybrain import autonomous_hexapod_episode as episode

    # Deliberately stimulate a disconnected DAN as a diagnostic source: no other
    # route can make it fire. Deliver odor at exactly the same step to expose loss.
    visual = VisualInterfaceMap(
        left_r1_r6_ids=(30,), right_r1_r6_ids=(100,),
        left_hs_ids=(101,), right_hs_ids=(102,),
        left_lc16_ids=(103,), right_lc16_ids=(104,),
    )
    monkeypatch.setattr(
        episode.VisualInterfaceEncoder, "encode_photoreceptor_spikes",
        lambda *args, **kwargs: (ExternalEvent(
            step=0, neuron_ids=(30,), voltages=(68.75,), channel="diagnostic",
        ),),
    )
    monkeypatch.setattr(
        episode, "poisson_voltage_events",
        lambda indices, **kwargs: {0: (indices, np.full(indices.size, 68.75))},
    )
    settings = config().model_copy(update={"visual_interface": visual})
    normal = run_autonomous_hexapod_episode(
        graph(), binding(), settings, **episode_kwargs(),
    )
    lesion = run_autonomous_hexapod_episode(
        graph(), binding(), settings, dan_enabled=False, **episode_kwargs(),
    )
    assert normal.dan_spike_counts == {30: 2}
    assert lesion.dan_spike_counts == {30: 0}
    assert normal.replay_exact and lesion.replay_exact
