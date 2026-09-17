from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from flybrain.biological_registry import load_biological_registry, resolve_biological_registry
from flybrain.graph import EventConnectome
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import (
    ResolvedPlasticEdgeManifest,
    resolve_plastic_edge_manifests,
)

REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
RETAINED_MALE_CNS = Path(
    "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5"
)


def test_resolves_only_exact_signed_plastic_and_modulatory_edges() -> None:
    if not RETAINED_MALE_CNS.is_dir():
        pytest.skip(f"retained snapshot unavailable: {RETAINED_MALE_CNS}")

    registry = load_biological_registry(REGISTRY)
    populations = resolve_biological_registry(registry, RETAINED_MALE_CNS)
    manifests = resolve_plastic_edge_manifests(registry, populations, RETAINED_MALE_CNS)

    by_name = {manifest.name: manifest for manifest in manifests}
    assert set(by_name) == {"kc_to_mbon", "dan_to_kc", "dan_to_mbon"}
    assert by_name["kc_to_mbon"].edge_count == 33_496
    assert by_name["kc_to_mbon"].contact_count == 402_850
    assert by_name["kc_to_mbon"].pre_count == 4_022
    assert by_name["kc_to_mbon"].post_count == 91
    assert by_name["kc_to_mbon"].sign == 1
    assert by_name["dan_to_kc"].edge_count == 5_624
    assert by_name["dan_to_kc"].contact_count == 36_149
    assert by_name["dan_to_kc"].sign == 0
    assert by_name["dan_to_mbon"].edge_count == 1_408
    assert by_name["dan_to_mbon"].contact_count == 36_583
    assert by_name["dan_to_mbon"].sign == 0


def test_binds_only_declared_manifest_pairs_to_canonical_csr_indices() -> None:
    graph = EventConnectome(
        neuron_ids=np.array([10, 20, 30], dtype=np.uint64),
        cell_types=("kc", "mbon", "other"),
        roles=("learning_kc", "learning_mbon", "interneuron"),
        transmitters=("acetylcholine",) * 3,
        superclasses=("central",) * 3,
        outgoing=csr_array(
            (np.array([4.0, 9.0], dtype=np.float32), ([0, 0], [1, 2])), shape=(3, 3)
        ),
    )
    manifest = ResolvedPlasticEdgeManifest(
        name="kc_to_mbon",
        sign=1,
        edge_count=1,
        contact_count=4,
        pre_count=1,
        post_count=1,
        edge_sha256="a" * 64,
        edge_pairs=((10, 20),),
    )

    binding = bind_manifest_to_graph(graph, manifest)

    assert binding.overlay.edge_indices.tolist() == [0]
    assert binding.pre_ids.tolist() == [10]
    assert binding.post_ids.tolist() == [20]
