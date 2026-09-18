import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.dynamics_stability_audit import audit_perturbation_recovery
from flybrain.graph import EventConnectome
from flybrain.shiu import ShiuParameters


def graph(weight: float) -> EventConnectome:
    return EventConnectome(
        neuron_ids=np.array([1], dtype=np.uint64),
        cell_types=("fixture",),
        roles=("fixture",),
        transmitters=("acetylcholine",),
        superclasses=("fixture",),
        outgoing=csr_array(np.array([[weight]], dtype=np.float32)),
    )


def test_audit_distinguishes_recovered_from_persistent_response() -> None:
    recovered = audit_perturbation_recovery(
        graph(0.0),
        stimulus_ids=(1,),
        observed_ids=(1,),
        params=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=3.0),
        stimulus_voltage_mv=8.0,
        window_steps=20,
        recovery_windows=2,
    )
    persistent = audit_perturbation_recovery(
        graph(200.0),
        stimulus_ids=(1,),
        observed_ids=(1,),
        params=ShiuParameters(dt_ms=0.5, refractory_ms=2.0, synaptic_delay_ms=3.0),
        stimulus_voltage_mv=8.0,
        window_steps=20,
        recovery_windows=2,
    )

    assert recovered.classification == "recovered"
    assert recovered.window_spikes == (1, 0, 0)
    assert persistent.classification == "persistent_activity"
    assert persistent.window_spikes[-1] > 0
    assert persistent.replay_exact is True
    assert persistent.graph_unchanged is True


def test_retained_malecns_olfactory_perturbation_is_not_yet_recovered() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    snapshot = Path(raw_snapshot)
    populations = resolve_biological_registry(
        load_biological_registry(
            Path("data/registry/autonomous-learning-registry-v1.json")
        ),
        snapshot,
    )
    from flybrain.graph import SparseConnectome

    graph_real = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    result = audit_perturbation_recovery(
        graph_real,
        stimulus_ids=populations.population("olfactory_sensory").neuron_ids,
        observed_ids=populations.population("kenyon_cells").neuron_ids,
        params=ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0),
        stimulus_voltage_mv=100.0,
        window_steps=100,
        recovery_windows=10,
        stimulus_mode="source_poisson",
    )

    assert result.evidence_kind == "simulation_observation"
    assert result.classification in {"recovered", "persistent_activity"}
    assert result.window_spikes[0] > 0
    assert result.window_spikes[-1] > 0
    assert result.replay_exact is True
    assert result.graph_unchanged is True
