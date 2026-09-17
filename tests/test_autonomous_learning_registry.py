from pathlib import Path

import pytest

from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)

REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
RETAINED_MALE_CNS = Path(
    "/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5"
)


def test_learning_registry_declares_complete_and_valence_bounded_populations() -> None:
    registry = load_biological_registry(REGISTRY)
    by_name = {population.name: population for population in registry.populations}

    assert set(by_name) == {
        "kenyon_cells",
        "mbons",
        "dan_appetitive_pam",
        "dan_aversive_ppl1",
        "dan_unassigned_ppl2",
    }
    assert by_name["kenyon_cells"].role == "learning_kc"
    assert by_name["mbons"].role == "learning_mbon"
    assert by_name["dan_appetitive_pam"].role == "dan_appetitive"
    assert by_name["dan_aversive_ppl1"].role == "dan_aversive"
    assert by_name["dan_unassigned_ppl2"].role == "dan_unassigned"
    assert by_name["kenyon_cells"].selector.expected_count == 4_064
    assert by_name["mbons"].selector.expected_count == 97
    assert by_name["dan_appetitive_pam"].selector.expected_count == 316
    assert by_name["dan_aversive_ppl1"].selector.expected_count == 16
    assert by_name["dan_unassigned_ppl2"].selector.expected_count == 8
    assert by_name["dan_unassigned_ppl2"].selector.in_values["type"] == (
        "PPL201",
        "PPL202",
        "PPL203",
        "PPL204",
    )
    assert by_name["kenyon_cells"].selector.expected_id_sha256 == (
        "50be316bdb907707ca51ea4baf2cba65f5fdbcf010f3bc38defef0fcfca6cc9d"
    )
    assert by_name["mbons"].selector.expected_id_sha256 == (
        "7c62a4af692408193e754422ec0bd803a25e63c94e3ce3850bba913ccfb45201"
    )


def test_learning_registry_resolves_full_retained_inventory() -> None:
    if not RETAINED_MALE_CNS.is_dir():
        pytest.skip(f"retained snapshot unavailable: {RETAINED_MALE_CNS}")

    resolved = resolve_biological_registry(
        load_biological_registry(REGISTRY), RETAINED_MALE_CNS
    )

    assert len(resolved.population("kenyon_cells").neuron_ids) == 4_064
    assert len(resolved.population("mbons").neuron_ids) == 97
    assert len(resolved.population("dan_appetitive_pam").neuron_ids) == 316
    assert len(resolved.population("dan_aversive_ppl1").neuron_ids) == 16
    assert len(resolved.population("dan_unassigned_ppl2").neuron_ids) == 8

    population_ids = [set(population.neuron_ids) for population in resolved.populations]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(population_ids)
        for right in population_ids[index + 1 :]
    )
