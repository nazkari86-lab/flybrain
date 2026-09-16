import hashlib

import numpy as np
from scipy.sparse import csr_array

from flybrain.biological_registry import AnnotationSelector, ResolvedPopulation, ResolvedRegistry
from flybrain.graph import EventConnectome
from flybrain.steering_benchmark import run_causal_steering_benchmark


def calibration_fixture() -> tuple[EventConnectome, ResolvedRegistry]:
    ids = np.array([1, 2, 3, 4, 5, 6, 10, 11, 20, 21, 30, 31], dtype=np.uint64)
    graph = EventConnectome(
        neuron_ids=ids,
        cell_types=("fixture",) * len(ids),
        roles=("interneuron",) * len(ids),
        transmitters=("acetylcholine",) * len(ids),
        superclasses=("fixture",) * len(ids),
        outgoing=csr_array((12, 12), dtype=np.float32),
    )
    names_and_ids = (
        ("visual_left", (1,)), ("visual_right", (2,)), ("hs_left", (3,)),
        ("hs_right", (4,)), ("lc16_left", (5,)), ("lc16_right", (6,)),
        ("d_na02_left", (10,)), ("d_na02_right", (11,)),
        ("d_ng13_left", (20,)), ("d_ng13_right", (21,)),
        ("mdn_left", (30,)), ("mdn_right", (31,)),
    )
    populations = tuple(
        ResolvedPopulation(
            name=name,
            role="sensory" if name.startswith(("visual", "hs", "lc16")) else "steering",
            selector=AnnotationSelector(equals={"type": "fixture"}),
            neuron_ids=values,
            id_sha256=hashlib.sha256(
                b"\n".join(str(value).encode() for value in values)
            ).hexdigest(),
            evidence=(),
        )
        for name, values in names_and_ids
    )
    registry = ResolvedRegistry(
        registry_version="fixture",
        registry_sha256="a" * 64,
        dataset_id="fixture",
        source_manifest_sha256="b" * 64,
        snapshot_content_sha256="c" * 64,
        annotation_sha256="d" * 64,
        populations=populations,
    )
    return graph, registry


def test_dn_calibration_reverses_lesions_restores_and_replays() -> None:
    graph, registry = calibration_fixture()
    result = run_causal_steering_benchmark(graph, registry, steps=12, seed=7)
    by_name = {condition.name: condition for condition in result.conditions}
    assert by_name["d_na02_left"].turn_integral > 0
    assert by_name["d_na02_right"].turn_integral < 0
    assert by_name["d_na02_left_silenced"].turn_integral == 0
    assert by_name["d_na02_left_restored"].trace_digest == by_name["d_na02_left"].trace_digest
    assert by_name["d_ng13_left"].turn_integral > 0
    assert by_name["d_ng13_right"].turn_integral < 0
    assert by_name["d_na02_bilateral"].turn_integral == 0
    assert by_name["mdn_bilateral"].reverse_integral > 0
    assert by_name["mdn_bilateral_silenced"].reverse_integral == 0
    assert by_name["mdn_left"].turn_integral == -by_name["mdn_right"].turn_integral
    assert by_name["d_na02_left"].relevant_spike_counts["d_na02_left"] > 0
    assert result.replay_exact is True
    assert result.graph_unchanged is True
    assert all(result.calibration_gates.values())
    assert result.calibration_passed is True


def test_benchmark_rejects_missing_declared_population() -> None:
    graph, registry = calibration_fixture()
    broken = registry.model_copy(
        update={
            "populations": tuple(
                item.model_copy(update={"neuron_ids": (999,)})
                if item.name == "mdn_right" else item
                for item in registry.populations
            )
        }
    )
    import pytest

    with pytest.raises(ValueError, match="absent from graph"):
        run_causal_steering_benchmark(graph, broken, steps=2, seed=1)
