import pytest

from flybrain.hexapod_backend import (
    ReferenceHexapodBackend,
    run_backend_trace,
)
from flybrain.hexapod_body import HexapodParameters, HexapodTorque


def test_reference_backend_replays_identical_torque_observation_trace() -> None:
    parameters = HexapodParameters(dt_s=0.001)
    schedule = (
        HexapodTorque.for_leg("left_fore", (0.001, 0.002, -0.002)),
        HexapodTorque.zero(),
        HexapodTorque.for_leg("right_fore", (-0.001, 0.002, -0.002)),
    )

    first = run_backend_trace(ReferenceHexapodBackend(parameters), schedule)
    second = run_backend_trace(ReferenceHexapodBackend(parameters), schedule)

    assert first.backend.name == "reference_hexapod"
    assert first.backend.evidence_kind == "model_assumption"
    assert first.backend.dt_s == 0.001
    assert first.torques == schedule
    assert first.observations == second.observations
    assert first.trace_digest == second.trace_digest
    assert first.observations[-1].time_s == pytest.approx(0.003)


def test_reference_backend_reset_restores_exact_initial_state() -> None:
    backend = ReferenceHexapodBackend()
    initial = backend.observe()
    backend.step(HexapodTorque.for_leg("left_fore", (0.001, 0.002, -0.002)))

    restored = backend.reset()

    assert restored == initial
    assert backend.observe() == initial


def test_backend_contract_rejects_torque_outside_declared_authority() -> None:
    backend = ReferenceHexapodBackend()

    with pytest.raises(ValueError, match="exceeds backend authority"):
        backend.step(HexapodTorque.for_leg("left_fore", (0.021, 0.0, 0.0)))


def test_backend_results_expose_no_task_or_policy_fields() -> None:
    result = run_backend_trace(
        ReferenceHexapodBackend(),
        (HexapodTorque.zero(),),
    )

    forbidden = {"reward", "target", "desired_action", "object_identity", "policy"}
    assert forbidden.isdisjoint(type(result).model_fields)
    assert forbidden.isdisjoint(type(result.observations[0]).model_fields)
