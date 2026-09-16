import math

import pytest

from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody, MotorCommand


def make_world(*, x: float = 1.0, y: float = 1.0, heading: float = 0.0) -> ArenaWorld:
    return ArenaWorld(
        ArenaConfig(width=10.0, height=10.0, dt_s=0.1, wall_restitution=0.2),
        FlyBody(x, y, heading, 0.0, 0.0, 1.0, (False,) * 6),
        food=(2.0, 1.0),
        threat=(8.0, 8.0),
    )


def test_forward_command_moves_body_and_consumes_energy() -> None:
    world = make_world()

    result = world.step(MotorCommand(forward=1.0, turn=0.0))

    assert result.body.x > 1.0
    assert result.body.energy < 1.0
    assert result.food_contact is False
    assert result.threat_contact is False


def test_wall_collision_keeps_body_inside_arena() -> None:
    world = make_world(x=9.99)

    result = world.step(MotorCommand(forward=1.0, turn=0.0))

    assert 0.0 <= result.body.x <= 10.0
    assert result.wall_contact is True
    assert result.body.forward_speed <= 0.0
    assert world.observe().wall_contact is True


def test_internal_wall_blocks_crossing_and_reverses_velocity() -> None:
    world = ArenaWorld(
        ArenaConfig(width=10.0, height=10.0, dt_s=0.1, wall_restitution=0.2),
        FlyBody(4.6, 5.0, 0.0, 3.0, 0.0, 1.0, (False,) * 6),
        food=(8.0, 8.0),
        threat=(9.0, 9.0),
        wall_segments=((5.0, 4.0, 5.0, 6.0),),
    )

    result = world.step(MotorCommand(forward=0.0, turn=0.0))

    assert result.wall_contact is True
    assert result.body.x <= 5.0 - world.contact_radius
    assert result.body.forward_speed < 0.0


def test_contacts_are_reported_from_world_geometry() -> None:
    world = make_world(x=2.0)

    result = world.step(MotorCommand(forward=0.0, turn=0.0))

    assert result.food_contact is True
    assert result.threat_contact is False


def test_commands_and_config_reject_non_finite_or_unbounded_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        MotorCommand(forward=math.inf, turn=0.0)
    with pytest.raises(ValueError, match="between"):
        MotorCommand(forward=2.0, turn=0.0)
    with pytest.raises(ValueError, match="width"):
        ArenaConfig(width=0.0, height=10.0, dt_s=0.1, wall_restitution=0.2)
    with pytest.raises(ValueError, match="axis-aligned"):
        ArenaWorld(
            ArenaConfig(10.0, 10.0, 0.1, 0.2),
            FlyBody(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
            food=(2.0, 2.0),
            threat=(8.0, 8.0),
            wall_segments=((1.0, 1.0, 2.0, 2.0),),
        )
