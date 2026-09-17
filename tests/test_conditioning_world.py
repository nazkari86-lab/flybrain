import pytest

from flybrain.conditioning_world import ConditioningSchedule


def test_schedule_is_deterministic_and_exposes_only_anonymous_sensory_events() -> None:
    first = ConditioningSchedule.create(seed=7, steps=12, appetitive_pair_steps=(2, 6))
    second = ConditioningSchedule.create(seed=7, steps=12, appetitive_pair_steps=(2, 6))

    assert first.digest == second.digest
    assert first.events[2].odor_intensity > 0.0
    assert first.events[2].appetitive_contact_intensity > 0.0
    forbidden = {"target", "object", "position", "reward", "action", "heading"}
    assert forbidden.isdisjoint(type(first.events[0]).model_fields)


def test_schedule_rejects_mixed_or_out_of_range_pairings() -> None:
    with pytest.raises(ValueError, match="both valences"):
        ConditioningSchedule.create(
            seed=7, steps=2, appetitive_pair_steps=(0,), aversive_pair_steps=(0,)
        )
    with pytest.raises(ValueError, match="outside schedule"):
        ConditioningSchedule.create(seed=7, steps=2, appetitive_pair_steps=(2,))
