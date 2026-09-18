"""Measured MBON-to-motor anatomical pathway audit without synthetic shortcuts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from flybrain.hexapod_motor import HexapodMotorMap


class MbonMotorRoute(BaseModel, frozen=True):
    """One exact two-synapse MBON-to-motor route from the retained dataset."""

    model_config = ConfigDict(extra="forbid")

    mbon_id: int = Field(gt=0)
    intermediate_id: int = Field(gt=0)
    motor_id: int = Field(gt=0)
    first_contacts: int = Field(gt=0)
    second_contacts: int = Field(gt=0)
    signs: tuple[Literal[-1, 0, 1], Literal[-1, 0, 1]]


class MbonMotorPathwayAudit(BaseModel, frozen=True):
    """Dataset measurement separating direct absence from retained two-hop anatomy."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: Literal["dataset_measurement"] = "dataset_measurement"
    direct_edge_count: int = Field(ge=0)
    direct_contact_count: int = Field(ge=0)
    two_hop_edge_count: int = Field(ge=0)
    two_hop_route_count: int = Field(ge=0)
    two_hop_intermediate_count: int = Field(ge=0)
    two_hop_contact_count: int = Field(ge=0)
    routes: tuple[MbonMotorRoute, ...]
    route_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def _digest(routes: tuple[MbonMotorRoute, ...]) -> str:
    payload = [route.model_dump(mode="json") for route in routes]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _validated_ids(mbon_ids: tuple[int, ...]) -> frozenset[int]:
    if not mbon_ids or any(type(item) is not int or item <= 0 for item in mbon_ids):
        raise ValueError("MBON IDs must be positive integers")
    if len(set(mbon_ids)) != len(mbon_ids):
        raise ValueError("MBON IDs must be unique")
    return frozenset(mbon_ids)


def _sign(value: int) -> Literal[-1, 0, 1]:
    parsed = value
    if parsed not in {-1, 0, 1}:
        raise ValueError(f"edge sign must be -1, 0, or 1: {parsed}")
    return cast(Literal[-1, 0, 1], parsed)


def resolve_mbon_motor_pathway(
    snapshot: Path,
    *,
    mbon_ids: tuple[int, ...],
    motor: HexapodMotorMap,
) -> MbonMotorPathwayAudit:
    """Resolve direct and exactly two-hop anatomy; never infer unmeasured edges."""

    mbon = _validated_ids(mbon_ids)
    motor_ids = {
        neuron_id for group in motor.groups for neuron_id in group.neuron_ids
    }
    table = pq.read_table(
        snapshot / "edges.parquet",
        columns=["pre_id", "post_id", "synapse_count", "sign"],
    )
    first_by_post: dict[int, list[tuple[int, int, Literal[-1, 0, 1]]]] = {}
    direct_contacts = 0
    direct_edges = 0
    raw_edges = zip(
        table.column("pre_id").to_pylist(),
        table.column("post_id").to_pylist(),
        table.column("synapse_count").to_pylist(),
        table.column("sign").to_pylist(),
        strict=True,
    )
    for pre, post, contacts, sign in raw_edges:
        pre_id, post_id = int(pre), int(post)
        contact_count, edge_sign = int(contacts), _sign(sign)
        if pre_id not in mbon:
            continue
        if post_id in motor_ids:
            direct_edges += 1
            direct_contacts += contact_count
        first_by_post.setdefault(post_id, []).append(
            (pre_id, contact_count, edge_sign)
        )

    routes: list[MbonMotorRoute] = []
    two_hop_edges: set[tuple[int, int]] = set()
    two_hop_contacts = 0
    raw_edges = zip(
        table.column("pre_id").to_pylist(),
        table.column("post_id").to_pylist(),
        table.column("synapse_count").to_pylist(),
        table.column("sign").to_pylist(),
        strict=True,
    )
    for pre, post, contacts, sign in raw_edges:
        intermediate_id, motor_id = int(pre), int(post)
        if intermediate_id not in first_by_post or motor_id not in motor_ids:
            continue
        contact_count, edge_sign = int(contacts), _sign(sign)
        two_hop_edges.add((intermediate_id, motor_id))
        two_hop_contacts += contact_count
        for mbon_id, first_contacts, first_sign in first_by_post[intermediate_id]:
            routes.append(
                MbonMotorRoute(
                    mbon_id=mbon_id,
                    intermediate_id=intermediate_id,
                    motor_id=motor_id,
                    first_contacts=first_contacts,
                    second_contacts=contact_count,
                    signs=(first_sign, edge_sign),
                )
            )
    resolved_routes = tuple(
        sorted(
            routes,
            key=lambda item: (item.mbon_id, item.intermediate_id, item.motor_id),
        )
    )
    return MbonMotorPathwayAudit(
        direct_edge_count=direct_edges,
        direct_contact_count=direct_contacts,
        two_hop_edge_count=len(two_hop_edges),
        two_hop_route_count=len(resolved_routes),
        two_hop_intermediate_count=len({item.intermediate_id for item in resolved_routes}),
        two_hop_contact_count=two_hop_contacts,
        routes=resolved_routes,
        route_digest=_digest(resolved_routes),
    )
