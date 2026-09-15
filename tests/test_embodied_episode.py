import numpy as np
from scipy.sparse import csr_array

from flybrain.embodied_episode import EmbodiedEpisodeConfig, run_embodied_episode
from flybrain.embodied_interfaces import MotorMap, SensoryMap
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.graph import EventConnectome, SparseConnectome


def fixture_graph() -> EventConnectome:
    return EventConnectome.from_sparse(
        SparseConnectome(
            neuron_ids=np.array([1, 2, 3, 4], dtype=np.uint64),
            cell_types=("sensory", "sensory", "motor", "interneuron"),
            roles=("sensory", "sensory", "motor", "interneuron"),
            adjacency=csr_array(
                (np.array([40.0, 40.0], dtype=np.float32), ([2, 2], [0, 1])),
                shape=(4, 4),
            ),
            transmitters=("acetylcholine",) * 4,
            superclasses=("central",) * 4,
        )
    )


def config() -> EmbodiedEpisodeConfig:
    return EmbodiedEpisodeConfig(
        max_steps=30,
        chunk_steps=1,
        seed=7,
        sensory_map=SensoryMap((1,), (2,), (), ()),
        motor_map=MotorMap((), (), (3,)),
    )


def world() -> ArenaWorld:
    return ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(1.0, 1.0),
        threat=(9.0, 9.0),
    )


def test_episode_closes_sensory_motor_world_loop_and_replays() -> None:
    graph = fixture_graph()

    first = run_embodied_episode(graph, config(), world=world())
    second = run_embodied_episode(graph, config(), world=world())

    assert first.passed is True
    assert first.replay_exact is True
    assert first.graph_unchanged is True
    assert len(first.sensory_events) > 0
    assert len(first.actions) == 30
    assert any(command.forward > 0 for command in first.actions)
    assert first.body_trace[-1].x > first.body_trace[0].x
    assert first.trace_digest == second.trace_digest


def test_silencing_motor_population_removes_world_action() -> None:
    graph = fixture_graph()

    active = run_embodied_episode(graph, config(), world=world())
    silenced = run_embodied_episode(graph, config(), world=world(), silence_motor=True)

    assert any(command.forward > 0 for command in active.actions)
    assert all(command.forward == 0 and command.turn == 0 for command in silenced.actions)
    assert active.body_trace[-1].x > silenced.body_trace[-1].x
