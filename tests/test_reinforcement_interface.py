import pytest

from flybrain.biological_registry import ResolvedPopulation, ResolvedRegistry
from flybrain.reinforcement_interface import AnonymousContact, ReinforcementInterface


def test_contact_interface_exposes_no_privileged_task_or_action_state() -> None:
    forbidden = {
        "reward",
        "target",
        "object",
        "distance",
        "heading",
        "desired_action",
        "action",
    }
    assert forbidden.isdisjoint(AnonymousContact.model_fields)

    interface = ReinforcementInterface(
        appetitive_dan_ids=(10, 11), aversive_dan_ids=(20,)
    )
    appetitive = interface.recruit(AnonymousContact(appetitive_intensity=0.7))
    aversive = interface.recruit(AnonymousContact(aversive_intensity=0.7))

    assert appetitive.dan_ids == (10, 11)
    assert aversive.dan_ids == (20,)
    assert appetitive.intensity == 0.7
    assert aversive.intensity == 0.7


def test_silent_contact_cannot_recruit_dan_and_mixed_valence_is_rejected() -> None:
    interface = ReinforcementInterface(appetitive_dan_ids=(10,), aversive_dan_ids=(20,))

    assert interface.recruit(AnonymousContact()).model_dump() == {
        "dan_ids": (),
        "intensity": 0.0,
    }
    with pytest.raises(ValueError, match="both DAN valences"):
        AnonymousContact(appetitive_intensity=0.1, aversive_intensity=0.1)


def test_interface_binds_only_declared_appetitive_and_aversive_registry_roles() -> None:
    def population(name: str, role: str, ids: tuple[int, ...]) -> ResolvedPopulation:
        return ResolvedPopulation(
            name=name,
            role=role,
            selector={"equals": {"class": "DAN"}},
            neuron_ids=ids,
            id_sha256="a" * 64,
            evidence=(),
        )

    registry = ResolvedRegistry(
        registry_version="fixture",
        registry_sha256="a" * 64,
        dataset_id="fixture",
        source_manifest_sha256="a" * 64,
        snapshot_content_sha256="a" * 64,
        annotation_sha256="a" * 64,
        populations=(
            population("pam", "dan_appetitive", (10,)),
            population("ppl1", "dan_aversive", (20,)),
            population("ppl2", "dan_unassigned", (30,)),
        ),
    )

    interface = ReinforcementInterface.from_resolved_registry(registry)

    assert interface.appetitive_dan_ids == (10,)
    assert interface.aversive_dan_ids == (20,)
