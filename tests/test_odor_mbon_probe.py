import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array
from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.graph import EventConnectome
from flybrain.odor_mbon_probe import (
    run_odor_mbon_condition,
    run_retained_odor_mbon_probe,
)
from flybrain.plastic_edge_binding import PlasticEdgeBinding
from flybrain.plastic_overlay import PlasticWeightOverlay
from flybrain.shiu import ShiuParameters


def _fixture() -> tuple[EventConnectome, PlasticEdgeBinding]:
    ids = np.asarray((1, 2, 3, 4), dtype=np.uint64)
    edges = ((1, 2, 100.0), (2, 3, 10.0), (2, 4, 10.0))
    index = {int(value): position for position, value in enumerate(ids)}
    graph = EventConnectome(
        neuron_ids=ids,
        cell_types=("source", "KC", "MBON", "MBON"),
        roles=("sensory", "learning_kc", "learning_mbon", "learning_mbon"),
        transmitters=("acetylcholine",) * 4,
        superclasses=("fixture",) * 4,
        outgoing=csr_array(
            (
                np.asarray([weight for _, _, weight in edges], dtype=np.float32),
                (
                    [index[pre] for pre, _, _ in edges],
                    [index[post] for _, post, _ in edges],
                ),
            ),
            shape=(4, 4),
        ),
    )
    edge_index = int(graph.outgoing.indptr[index[2]])
    binding = PlasticEdgeBinding(
        overlay=PlasticWeightOverlay.create(
            edge_indices=np.asarray([edge_index], dtype=np.int64),
            canonical_edge_count=graph.edge_count,
        ),
        pre_ids=np.asarray([2], dtype=np.uint64),
        post_ids=np.asarray([3], dtype=np.uint64),
    )
    return graph, binding


def test_max2_uses_same_source_events_and_changes_only_declared_target_input() -> None:
    graph, binding = _fixture()
    canonical = graph.outgoing.data.copy()
    parameters = ShiuParameters(
        dt_ms=0.1,
        refractory_ms=2.0,
        synaptic_delay_ms=1.0,
        poisson_rate_hz=10_000.0,
    )
    kwargs = dict(
        source_ids=(1,),
        target_ids=(3, 4),
        steps=200,
        seed=7,
        parameters=parameters,
    )

    baseline = run_odor_mbon_condition(graph, binding, name="threat", **kwargs)
    replay = run_odor_mbon_condition(graph, binding, name="threat", **kwargs)
    boosted = run_odor_mbon_condition(
        graph, binding, name="threat_max2", max2_target_id=3, **kwargs
    )

    assert baseline.source_voltage_events == boosted.source_voltage_events == 200
    assert baseline.source_event_digest == boosted.source_event_digest
    assert baseline.neural_trace_digest == replay.neural_trace_digest
    assert baseline.source_spikes > 0
    assert baseline.kc_input_spike_counts[3] > 0
    assert boosted.kc_input_spike_counts == baseline.kc_input_spike_counts
    assert (
        boosted.effective_positive_weighted_spikes[3]
        == 2 * baseline.effective_positive_weighted_spikes[3]
    )
    assert (
        boosted.effective_positive_weighted_spikes[4]
        == baseline.effective_positive_weighted_spikes[4]
    )
    assert boosted.artificial_intervention == "max2_kc_to_mbon_3"
    assert np.array_equal(graph.outgoing.data, canonical)
    assert np.array_equal(binding.overlay.multipliers, np.ones(1, dtype=np.float32))


def test_no_source_stays_inactive_and_unknown_target_fails_closed() -> None:
    graph, binding = _fixture()
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)

    quiet = run_odor_mbon_condition(
        graph,
        binding,
        name="none",
        source_ids=(),
        target_ids=(3, 4),
        steps=40,
        seed=7,
        parameters=parameters,
    )

    assert quiet.source_voltage_events == 0
    assert quiet.mbon_spike_counts == {3: 0, 4: 0}
    assert quiet.kc_input_spike_counts == {3: 0, 4: 0}
    with pytest.raises(ValueError, match="target"):
        run_odor_mbon_condition(
            graph,
            binding,
            name="missing",
            source_ids=(1,),
            target_ids=(3, 9),
            steps=40,
            seed=7,
            parameters=parameters,
        )


def test_source_id_order_does_not_change_poisson_assignment() -> None:
    graph, binding = _fixture()
    parameters = ShiuParameters(dt_ms=0.1, refractory_ms=2.0, synaptic_delay_ms=1.0)

    forward = run_odor_mbon_condition(
        graph,
        binding,
        name="paired",
        source_ids=(1, 2),
        target_ids=(3, 4),
        steps=500,
        seed=7,
        parameters=parameters,
    )
    reverse = run_odor_mbon_condition(
        graph,
        binding,
        name="paired",
        source_ids=(2, 1),
        target_ids=(3, 4),
        steps=500,
        seed=7,
        parameters=parameters,
    )

    assert forward.source_event_digest == reverse.source_event_digest
    assert forward.neural_trace_digest == reverse.neural_trace_digest


def test_retained_probe_resolves_real_odor_and_dan_routes_without_behavior_claim() -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")

    result = run_retained_odor_mbon_probe(Path(raw_snapshot), steps=20, seeds=(7,))

    assert result.protocol == "retained-odor-mbon-probe-v1"
    assert result.graph_neurons == 166_606
    assert result.graph_edges == 6_240_402
    assert len(result.source_ids["food"]) == 148
    assert len(result.source_ids["threat"]) == 41
    assert result.dan_route_counts[519128] == {"appetitive": 18, "aversive": 0}
    assert result.dan_route_counts[524893] == {"appetitive": 27, "aversive": 1}
    assert [item.name for item in result.conditions] == [
        "none",
        "food",
        "threat",
        "both",
        "threat_max2",
    ]
    assert result.conditions[2].source_voltage_events == result.conditions[4].source_voltage_events
    assert result.conditions[4].artificial_intervention == "max2_kc_to_mbon_519128"
    assert result.replay_exact is True
    assert result.paired_source_events_exact is True
    assert result.graph_unchanged is True
    assert result.overlay_unchanged is True
    assert result.autonomous_behavior_claim_allowed is False


def test_cli_refuses_to_overwrite_existing_result(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    output.write_text("preserve", encoding="utf-8")

    invocation = CliRunner().invoke(
        app,
        ["experiment", "odor-mbon", "missing-snapshot", "--output", str(output)],
    )

    assert invocation.exit_code != 0
    assert "output already exists" in invocation.output
    assert output.read_text(encoding="utf-8") == "preserve"


def test_cli_writes_real_replay_checked_result_atomically(tmp_path: Path) -> None:
    raw_snapshot = os.environ.get("FLYBRAIN_MALECNS_SNAPSHOT")
    if raw_snapshot is None:
        pytest.skip("real MaleCNS snapshot is not configured")
    output = tmp_path / "probe.json"

    invocation = CliRunner().invoke(
        app,
        [
            "experiment",
            "odor-mbon",
            raw_snapshot,
            "--output",
            str(output),
            "--steps",
            "20",
            "--seeds",
            "7",
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == json.loads(invocation.stdout)
    assert payload["protocol"] == "retained-odor-mbon-probe-v1"
    assert payload["replay_exact"] is True
    assert payload["graph_unchanged"] is True
    assert payload["autonomous_behavior_claim_allowed"] is False
