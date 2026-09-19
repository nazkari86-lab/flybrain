"""Strict torque/observation boundary for interchangeable hexapod physics."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from flybrain.hexapod_body import (
    HexapodBody,
    HexapodParameters,
    HexapodTorque,
    ReferenceHexapod,
)


class BackendIdentity(BaseModel, frozen=True):
    """Serialized backend provenance without behavioral claims."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    evidence_kind: Literal["model_assumption"] = "model_assumption"
    dt_s: float = Field(gt=0.0)


class HexapodBackend(Protocol):
    """The only physics interface available to the neural controller."""

    @property
    def identity(self) -> BackendIdentity: ...

    @property
    def parameters(self) -> HexapodParameters: ...

    def observe(self) -> HexapodBody: ...

    def step(self, torque: HexapodTorque) -> HexapodBody: ...

    def reset(self) -> HexapodBody: ...


class BackendTrace(BaseModel, frozen=True):
    """Replayable backend trace generated from an immutable torque schedule."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendIdentity
    torques: tuple[HexapodTorque, ...]
    observations: tuple[HexapodBody, ...]
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BackendParityResult(BaseModel, frozen=True):
    """Direction-only parity result; magnitudes remain backend-specific."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    reference_backend: BackendIdentity
    candidate_backend: BackendIdentity
    compared_joints: int = Field(ge=0, le=18)
    direction_matches: int = Field(ge=0, le=18)
    reference_replay_exact: bool
    candidate_replay_exact: bool
    identical_torque_schedule: bool


def _validate_torque(torque: HexapodTorque, parameters: HexapodParameters) -> None:
    for values in torque.values:
        if any(
            abs(value) > limit
            for value, limit in zip(values, parameters.max_torque_nm, strict=True)
        ):
            raise ValueError("requested torque exceeds backend authority")


class ReferenceHexapodBackend:
    """Deterministic reference implementation of the shared backend contract."""

    def __init__(self, parameters: HexapodParameters | None = None) -> None:
        self._parameters = parameters or HexapodParameters()
        self._body = ReferenceHexapod(self._parameters)

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(
            name="reference_hexapod",
            version="reference-hexapod-v1",
            dt_s=self._parameters.dt_s,
        )

    @property
    def parameters(self) -> HexapodParameters:
        return self._parameters

    def observe(self) -> HexapodBody:
        return self._body.observe()

    def step(self, torque: HexapodTorque) -> HexapodBody:
        _validate_torque(torque, self._parameters)
        return self._body.step(torque)

    def reset(self) -> HexapodBody:
        self._body = ReferenceHexapod(self._parameters)
        return self._body.observe()


def run_backend_trace(
    backend: HexapodBackend,
    torques: tuple[HexapodTorque, ...],
) -> BackendTrace:
    """Apply one predeclared torque schedule through the shared backend boundary."""

    observations = tuple(backend.step(torque) for torque in torques)
    payload = {
        "backend": backend.identity.model_dump(mode="json"),
        "torques": [torque.model_dump(mode="json") for torque in torques],
        "observations": [item.model_dump(mode="json") for item in observations],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BackendTrace(
        backend=backend.identity,
        torques=torques,
        observations=observations,
        trace_digest=digest,
    )


def _joint_deltas(initial: HexapodBody, final: HexapodBody) -> tuple[float, ...]:
    return tuple(
        after - before
        for initial_leg, final_leg in zip(initial.legs, final.legs, strict=True)
        for before, after in zip(
            initial_leg.joint_angles_rad,
            final_leg.joint_angles_rad,
            strict=True,
        )
    )


def run_backend_parity(
    reference: HexapodBackend,
    candidate: HexapodBackend,
    torques: tuple[HexapodTorque, ...],
    *,
    minimum_joint_response_rad: float = 1e-12,
) -> BackendParityResult:
    """Compare intervention direction under one identical, predeclared torque schedule."""

    if not torques:
        raise ValueError("backend parity requires a non-empty torque schedule")
    if minimum_joint_response_rad <= 0.0:
        raise ValueError("minimum joint response must be positive")
    reference_initial = reference.observe()
    candidate_initial = candidate.observe()
    reference_first = run_backend_trace(reference, torques)
    candidate_first = run_backend_trace(candidate, torques)
    reference.reset()
    candidate.reset()
    reference_replay = run_backend_trace(reference, torques)
    candidate_replay = run_backend_trace(candidate, torques)
    reference_deltas = _joint_deltas(
        reference_initial,
        reference_first.observations[-1],
    )
    candidate_deltas = _joint_deltas(
        candidate_initial,
        candidate_first.observations[-1],
    )
    compared = tuple(
        (reference_delta, candidate_delta)
        for reference_delta, candidate_delta in zip(
            reference_deltas,
            candidate_deltas,
            strict=True,
        )
        if abs(reference_delta) >= minimum_joint_response_rad
        and abs(candidate_delta) >= minimum_joint_response_rad
    )
    matches = sum(
        reference_delta * candidate_delta > 0.0
        for reference_delta, candidate_delta in compared
    )
    if not compared:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
    elif matches == len(compared) == 18:
        classification = "positive"
    elif matches == 0:
        classification = "directionally_wrong"
    else:
        classification = "null"
    return BackendParityResult(
        classification=classification,
        reference_backend=reference.identity,
        candidate_backend=candidate.identity,
        compared_joints=len(compared),
        direction_matches=matches,
        reference_replay_exact=(reference_first == reference_replay),
        candidate_replay_exact=(candidate_first == candidate_replay),
        identical_torque_schedule=(
            reference_first.torques == candidate_first.torques == torques
        ),
    )
