import numpy as np
from scipy.sparse import csr_array

from flybrain.dynamics_gain_calibration import sweep_synaptic_gain
from flybrain.graph import EventConnectome
from flybrain.shiu import ShiuParameters


def test_sweep_selects_first_gain_with_delayed_response_and_recovery() -> None:
    graph = EventConnectome(
        neuron_ids=np.array([1, 2], dtype=np.uint64),
        cell_types=("fixture", "fixture"),
        roles=("fixture", "fixture"),
        transmitters=("acetylcholine", "acetylcholine"),
        superclasses=("fixture", "fixture"),
        outgoing=csr_array(np.array([[0.0, 200.0], [0.0, 0.0]], dtype=np.float32)),
    )

    result = sweep_synaptic_gain(
        graph,
        stimulus_ids=(1,),
        observed_ids=(2,),
        params=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=3.0),
        candidate_synapse_mv=(0.1, 0.5, 1.0),
        stimulus_voltage_mv=8.0,
        window_steps=1,
        recovery_windows=20,
        stimulus_mode="direct_voltage",
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.selection_kind == "model_assumption"
    assert [point.accepted for point in result.points] == [False, True, True]
    assert result.selected_synapse_mv == 0.5
    assert result.points[1].audit.classification == "recovered"
    assert result.points[1].audit.replay_exact is True
    assert result.points[1].audit.graph_unchanged is True
