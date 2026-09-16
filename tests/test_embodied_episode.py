import numpy as np
from scipy.sparse import csr_array

from flybrain.embodied_episode import EmbodiedEpisodeConfig, run_embodied_episode
from flybrain.embodied_interfaces import MotorMap, SensoryMap
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.plasticity import PlasticEdgeSet


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


def test_chunk_steps_control_neural_time_per_world_step() -> None:
    fast = config().model_copy(update={"max_steps": 2, "chunk_steps": 20})
    slow = config().model_copy(update={"max_steps": 2, "chunk_steps": 1})

    fast_result = run_embodied_episode(fixture_graph(), fast, world=world())
    slow_result = run_embodied_episode(fixture_graph(), slow, world=world())

    assert fast_result.neural_steps == 40
    assert slow_result.neural_steps == 2
    assert any(command.forward > 0 for command in fast_result.actions)
    assert all(command.forward == 0 for command in slow_result.actions)


def test_world_reward_updates_a_copied_plastic_overlay() -> None:
    edges = PlasticEdgeSet.create(
        pre_ids=np.array([1], dtype=np.uint64),
        post_ids=np.array([3], dtype=np.uint64),
        baseline_weights=np.array([40.0], dtype=np.float32),
    )
    learned_config = config().model_copy(update={"max_steps": 2, "chunk_steps": 20})

    result = run_embodied_episode(
        fixture_graph(), learned_config, world=world(), plastic_edges=edges
    )

    assert result.learning_applied is True
    assert result.final_plastic_multipliers[0] < 1.0
    assert edges.multipliers.tolist() == [1.0]


def test_no_world_contact_emits_no_reward_or_dopamine() -> None:
    distant = ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(8.0, 8.0),
        threat=(9.0, 9.0),
    )

    result = run_embodied_episode(fixture_graph(), config(), world=distant)

    assert set(result.rewards) == {0.0}
    assert set(result.dopamine) == {0.0}
