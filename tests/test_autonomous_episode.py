import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.autonomous_episode import (
    AutonomousEpisodeConfig,
    run_autonomous_episode,
)
from flybrain.embodied_interfaces import MotorMap, SensoryMap
from flybrain.embodied_world import ArenaConfig, ArenaWorld, FlyBody
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.mushroom_body_learning import MushroomBodyLearningParameters
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters


def graph() -> EventConnectome:
    return EventConnectome.from_sparse(
        SparseConnectome(
            neuron_ids=np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            cell_types=("sensory", "Kenyon_Cell", "MBON", "DAN", "motor"),
            roles=("sensory", "learning_kc", "learning_mbon", "dan_appetitive", "motor"),
            adjacency=csr_array(
                (np.array([400.0, 200.0, 200.0, 40.0], dtype=np.float32),
                 ([1, 2, 4, 2], [0, 1, 2, 3])),
                shape=(5, 5),
            ),
            transmitters=("acetylcholine",) * 5,
            superclasses=("central",) * 5,
        )
    )


def binding() -> PlasticEdgeBinding:
    return PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.array([1], dtype=np.int64), canonical_edge_count=4
        ),
        pre_ids=np.array([2], dtype=np.uint64),
        post_ids=np.array([3], dtype=np.uint64),
    )


def config() -> AutonomousEpisodeConfig:
    return AutonomousEpisodeConfig(
        max_steps=2,
        chunk_steps=1000,
        seed=7,
        sensory_map=SensoryMap((1,), (), (), ()),
        motor_map=MotorMap((), (), (5,)),
        shiu_parameters=ShiuParameters(synapse_mv=0.275),
        learning_parameters=MushroomBodyLearningParameters(),
    )


def world() -> ArenaWorld:
    return ArenaWorld(
        ArenaConfig(10.0, 10.0, 0.1, 0.2),
        FlyBody(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, (False,) * 6),
        food=(1.0, 1.0),
        threat=(9.0, 9.0),
    )


def reinforcement() -> ReinforcementInterface:
    return ReinforcementInterface(appetitive_dan_ids=(4,), aversive_dan_ids=(5,))


def test_autonomous_episode_routes_contact_to_dan_and_plasticity() -> None:
    first = run_autonomous_episode(
        graph(),
        config(),
        binding=binding(),
        dan_to_mbon_pairs=((4, 3),),
        reinforcement=reinforcement(),
        world=world(),
    )

    assert first.replay_exact is True
    assert first.graph_unchanged is True
    assert first.contact_events == 2
    assert first.dan_events == 2
    assert first.motor_spikes > 0
    assert first.active_motor_ids == (5,)
    assert first.motor_population_coverage == 1.0
    assert first.all_declared_motors_active is True
    assert first.learning_applied is True
    assert first.final_multipliers[0] < 1.0


def test_autonomous_episode_rejects_neural_body_clock_mismatch() -> None:
    mismatched = config().model_copy(update={"chunk_steps": 20})

    with pytest.raises(ValueError, match="neural chunk duration"):
        run_autonomous_episode(
            graph(),
            mismatched,
            binding=binding(),
            dan_to_mbon_pairs=((4, 3),),
            reinforcement=reinforcement(),
            world=world(),
        )
