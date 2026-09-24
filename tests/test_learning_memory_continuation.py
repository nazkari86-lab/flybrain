"""Learning state must survive episode and serialization boundaries."""

import numpy as np
import pytest
from test_associative_motor_loop import binding, graph
from test_autonomous_evidence import benchmark_config, episode_kwargs
from test_autonomous_hexapod_episode import config

from flybrain.autonomous_hexapod_episode import run_autonomous_hexapod_episode


def train(state, settings, **kwargs):
    return run_autonomous_hexapod_episode(
        graph(), state, settings, mutate_binding=True, **episode_kwargs(), **kwargs
    )


@pytest.mark.parametrize("slow", [False, True])
def test_episode_and_json_boundaries_retain_fast_and_slow_learning(slow):
    settings = config().model_copy(update={"nitric_oxide_dan_ids": (30,) if slow else ()})
    state = binding()
    first = train(state, settings)
    assert hasattr(first, "learning_memory"), "episode discards fast and slow memory"
    memory = first.learning_memory
    assert max(memory.eligibility) > 0
    assert max(memory.dopamine_trace) > 0
    restored = type(memory).model_validate_json(memory.model_dump_json())
    assert restored == memory
    second = train(state, settings, learning_memory=restored)
    assert second.replay_exact
    assert second.learning_memory.eligibility[0] > memory.eligibility[0]
    assert second.learning_memory.dopamine_trace[0] > memory.dopamine_trace[0]
    if slow:
        assert second.slow_memory_dopamine_effect_max > first.slow_memory_dopamine_effect_max
        assert (
            second.slow_memory_nitric_oxide_effect_max > first.slow_memory_nitric_oxide_effect_max
        )

    frozen = run_autonomous_hexapod_episode(
        graph(), state, settings, learning_memory=second.learning_memory,
        plasticity_enabled=False, **episode_kwargs(),
    )
    assert frozen.replay_exact
    assert frozen.learning_memory == second.learning_memory
    assert frozen.final_multipliers == second.final_multipliers


def test_stale_weights_and_different_routes_cannot_reuse_learning_memory():
    state = binding()
    settings = config()
    first = train(state, settings)
    assert hasattr(first, "learning_memory"), "episode discards learning identity"
    with pytest.raises(ValueError, match="weights"):
        train(binding(), settings, learning_memory=first.learning_memory)
    changed = settings.model_copy(update={
        "learning": settings.learning.model_copy(update={"dan_to_mbon_pairs": ((100, 20),)}),
    })
    with pytest.raises(ValueError, match="identity"):
        train(state, changed, learning_memory=first.learning_memory)


def test_loaded_slow_state_is_not_multiplied_into_fast_weights_twice():
    settings = config().model_copy(update={"nitric_oxide_dan_ids": (30,)})
    state = binding()
    first = train(state, settings)
    assert hasattr(first, "learning_memory"), "episode discards separate fast weights"
    # The effective overlay contains slow modulation; fast multipliers do not.
    memory = first.learning_memory
    assert memory.base_multipliers != memory.effective_multipliers
    before = state.overlay.multipliers.copy()
    second = run_autonomous_hexapod_episode(
        graph(), state, settings, learning_memory=memory,
        plasticity_enabled=False, **episode_kwargs(),
    )
    np.testing.assert_array_equal(state.overlay.multipliers, before)
    assert second.learning_memory.base_multipliers == memory.base_multipliers
    assert second.final_multipliers == memory.effective_multipliers


def test_benchmark_threads_memory_only_within_each_condition_and_seed():
    from flybrain.autonomous_behavior_benchmark import run_behavior_benchmark

    values = benchmark_config().model_dump()
    values.update(training_episodes=2, seeds=(7, 11), nitric_oxide_dan_ids=(30,))
    result = run_behavior_benchmark(
        graph(), binding(), type(benchmark_config())(**values),
        learning=config().learning, **episode_kwargs(),
    )
    assert hasattr(result, "holdout_memory_frozen"), "benchmark drops continuation state"
    assert result.holdout_memory_frozen
    groups = {}
    for episode in result.episode_evidence:
        groups.setdefault((episode.condition, episode.replicate_seed), []).append(episode)
    assert len(groups) == 10
    for episodes in groups.values():
        assert episodes[0].initial_memory_digest is None
        assert episodes[1].initial_memory_digest == episodes[0].final_memory_digest
        assert episodes[2].initial_memory_digest == episodes[1].final_memory_digest
        assert episodes[2].memory_unchanged
        assert episodes[2].initial_memory_digest == episodes[2].final_memory_digest


def test_memory_checkpoint_refuses_overwrite_and_detects_corruption(tmp_path):
    import json

    from flybrain import learning_memory as persistence

    state = binding()
    first = train(state, config())
    assert hasattr(persistence, "save_learning_memory"), "memory has no durable checkpoint"
    path = tmp_path / "learning.json"
    persistence.save_learning_memory(path, first.learning_memory)
    loaded = persistence.load_learning_memory(path)
    assert loaded == first.learning_memory
    assert train(state, config(), learning_memory=loaded).replay_exact
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        persistence.save_learning_memory(path, loaded)
    assert path.read_bytes() == content
    tampered = json.loads(content)
    tampered["memory"]["eligibility"][0] += 1.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest"):
        persistence.load_learning_memory(path)
