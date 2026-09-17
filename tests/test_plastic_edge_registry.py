from pathlib import Path

import pytest

from flybrain.biological_registry import load_biological_registry, resolve_biological_registry
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests

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
