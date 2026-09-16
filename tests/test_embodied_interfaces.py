from flybrain.embodied_interfaces import MotorDecoder, MotorMap, SensoryEncoder, SensoryMap
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody, MotorCommand


def world_step(*, food: tuple[float, float] = (1.1, 1.0)):
    world = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=food,
        threat=(8.0, 8.0),
    )
    return world.step(MotorCommand(0.0, 0.0))


def test_sensor_encoding_is_deterministic_and_world_derived() -> None:
    sensor_map = SensoryMap((10, 11), (20,), (30,), (40, 41))
    encoder = SensoryEncoder(sensor_map, arena_width=10.0, arena_height=10.0)

    first = encoder.encode(world_step(), step=4)
    second = encoder.encode(world_step(), step=4)

    assert first == second
    assert {event.channel for event in first} == {"visual", "odor", "touch", "proprioception"}
    assert all(event.step == 4 and len(event.neuron_ids) == len(event.voltages) for event in first)
    assert any(event.channel == "touch" and event.voltages[0] > 0 for event in first)


def test_odor_strength_depends_on_food_relative_distance() -> None:
    encoder = SensoryEncoder(
        SensoryMap((), (20,), (), ()), arena_width=10.0, arena_height=10.0
    )

    near = encoder.encode(world_step(food=(1.5, 1.0)), step=0)[0]
    far = encoder.encode(world_step(food=(9.0, 9.0)), step=0)[0]

    assert near.voltages[0] > far.voltages[0]


def test_motor_decoder_votes_declared_output_populations() -> None:
    decoder = MotorDecoder(MotorMap(left_ids=(1, 2), right_ids=(3, 4), forward_ids=(5,)))

    command = decoder.decode((1, 2, 5))

    assert command == MotorCommand(forward=1.0, turn=1.0)
