"""Causal direct-motor and phase-gait calibration for the reference hexapod."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flybrain.graph import EventConnectome
from flybrain.hexapod_body import (
    JOINT_NAMES,
    LEG_NAMES,
    TRIPOD_A,
    TRIPOD_B,
    HexapodBody,
    HexapodParameters,
    ReferenceHexapod,
)
from flybrain.hexapod_motor import HexapodMotorDecoder, HexapodMotorMap, MotorGroup


class ProtocolName(StrEnum):
    MOTOR_GROUP_CALIBRATION = "motor_group_calibration"
    PHASE_GAIT_CALIBRATION = "phase_gait_calibration"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _graph_digest(graph: EventConnectome) -> str:
    digest = hashlib.sha256()
    for array in (
        graph.neuron_ids,
        graph.outgoing.data,
        graph.outgoing.indices,
        graph.outgoing.indptr,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


class MotorSpikeSchedule(BaseModel, frozen=True):
    """Prebuilt target-independent direct motor spike schedule."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    spikes_by_step: tuple[tuple[int, ...], ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if not self.spikes_by_step:
            raise ValueError("motor schedule must contain at least one step")
        if any(
            type(neuron_id) is not int or neuron_id <= 0
            for spikes in self.spikes_by_step
            for neuron_id in spikes
        ):
            raise ValueError("motor schedule must contain positive integer IDs")
        expected = _digest(
            {"name": self.name, "spikes_by_step": self.spikes_by_step}
        )
        if self.sha256 != expected:
            raise ValueError("motor schedule digest disagrees with its contents")
        return self

    @classmethod
    def build(
        cls,
        name: str,
        spikes_by_step: tuple[tuple[int, ...], ...],
    ) -> MotorSpikeSchedule:
        return cls(
            name=name,
            spikes_by_step=spikes_by_step,
            sha256=_digest({"name": name, "spikes_by_step": spikes_by_step}),
        )


class HexapodBenchmarkThresholds(BaseModel, frozen=True):
    """Predeclared direct-motor and gait classification thresholds."""

    model_config = ConfigDict(extra="forbid")

    minimum_motor_spikes: int = Field(default=1, gt=0)
    minimum_joint_motion_rad: float = Field(default=1e-6, gt=0.0)
    minimum_lesion_fraction: float = Field(default=0.9, ge=0.0, le=1.0)
    minimum_forward_displacement_m: float = Field(default=1e-5, gt=0.0)
    maximum_lateral_drift_m: float = Field(default=0.2, gt=0.0)
    minimum_support_margin_m: float = -0.02

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        values: tuple[float, ...] = (
            self.minimum_joint_motion_rad,
            self.minimum_lesion_fraction,
            self.minimum_forward_displacement_m,
            self.maximum_lateral_drift_m,
            self.minimum_support_margin_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("benchmark thresholds must be finite")
        return self


class BenchmarkAssumption(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    evidence_kind: Literal["model_assumption"] = "model_assumption"
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HexapodConditionResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: ProtocolName
    schedule_name: str
    schedule_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    spike_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    torque_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stimulated_population: str | None
    silenced_populations: tuple[str, ...]
    motor_spikes: int = Field(ge=0)
    joint_motion_rad: float | None
    forward_displacement_m: float
    lateral_drift_m: float = Field(ge=0.0)
    minimum_support_margin_m: float
    alternating_support: bool
    energy_j: float = Field(ge=0.0)
    fallen: bool
    final_body: HexapodBody

    @model_validator(mode="after")
    def validate_finite_metrics(self) -> Self:
        values: tuple[float, ...] = (
            self.forward_displacement_m,
            self.lateral_drift_m,
            self.minimum_support_margin_m,
            self.energy_j,
        )
        if self.joint_motion_rad is not None:
            values = (*values, self.joint_motion_rad)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("hexapod condition metrics must be finite")
        return self


class CausalComparison(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    name: str
    normal_condition: str
    intervention_condition: str
    metric: Literal["joint_motion_rad", "forward_displacement_m"]
    normal_value: float
    intervention_value: float
    lesion_fraction: float


class ClaimClassification(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["positive", "null", "directionally_wrong", "underpowered"]
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    numerator: float
    denominator: float
    threshold: float
    lesion_fraction: float
    reasons: tuple[str, ...]


class HexapodBenchmarkResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")

    benchmark: Literal["hexapod-motor-v1"] = "hexapod-motor-v1"
    graph_neurons: int
    graph_edges: int
    thresholds: HexapodBenchmarkThresholds
    assumptions: tuple[BenchmarkAssumption, ...]
    schedules: tuple[MotorSpikeSchedule, ...]
    conditions: tuple[HexapodConditionResult, ...]
    comparisons: tuple[CausalComparison, ...]
    direct_motor_claims: dict[str, ClaimClassification]
    gait_claim: ClaimClassification
    calibration_gates: dict[str, bool]
    calibration_passed: bool
    replay_exact: bool
    graph_unchanged: bool


def build_direct_motor_schedules(
    mapping: HexapodMotorMap,
    *,
    steps: int,
) -> tuple[MotorSpikeSchedule, ...]:
    """Build every direct schedule before any condition executes."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("motor schedule steps must be a positive integer")
    return tuple(
        MotorSpikeSchedule.build(
            name=f"direct:{group.name}",
            spikes_by_step=(group.neuron_ids,) * steps,
        )
        for group in mapping.groups
    )


def _build_phase_gait_schedule(
    mapping: HexapodMotorMap,
    parameters: HexapodParameters,
    *,
    steps: int,
) -> MotorSpikeSchedule:
    offsets = (0.0, 0.5, 0.5, 0.0, 0.0, 0.5)
    spike_steps = []
    for step in range(steps):
        spikes: list[int] = []
        for leg_index, leg in enumerate(LEG_NAMES):
            phase = (
                offsets[leg_index] + step * parameters.phase_rate_hz * parameters.dt_s
            ) % 1.0
            if phase < 0.5:
                names = (
                    f"{leg}_trochanter_flexor",
                    f"{leg}_tibia_extensor",
                )
            else:
                names = (
                    f"{leg}_trochanter_extensor",
                    f"{leg}_tibia_flexor",
                )
            for name in names:
                spikes.extend(mapping.group(name).neuron_ids)
        spike_steps.append(tuple(spikes))
    return MotorSpikeSchedule.build(
        name="phase_gait",
        spikes_by_step=tuple(spike_steps),
    )


def _silenced_ids(
    mapping: HexapodMotorMap,
    names: frozenset[str],
) -> frozenset[int]:
    populations = mapping.named_populations()
    unknown = sorted(names - populations.keys())
    if unknown:
        raise ValueError(f"unknown motor populations: {unknown}")
    return frozenset(
        neuron_id for name in names for neuron_id in populations[name]
    )


def _alternating_support(contact_sets: tuple[frozenset[str], ...]) -> bool:
    tripod_a = frozenset(TRIPOD_A)
    tripod_b = frozenset(TRIPOD_B)
    saw_a = any(contacts == tripod_a for contacts in contact_sets)
    saw_b = any(contacts == tripod_b for contacts in contact_sets)
    return saw_a and saw_b


def _run_condition(
    mapping: HexapodMotorMap,
    schedule: MotorSpikeSchedule,
    *,
    name: str,
    protocol: ProtocolName,
    parameters: HexapodParameters,
    silenced: frozenset[str] = frozenset(),
    target_group: MotorGroup | None = None,
) -> HexapodConditionResult:
    selected = _silenced_ids(mapping, silenced)
    simulator = ReferenceHexapod(parameters)
    decoder = HexapodMotorDecoder(mapping)
    initial = simulator.observe()
    initial_angle: float | None = None
    if target_group is not None:
        joint_index = JOINT_NAMES.index(target_group.joint)
        initial_angle = initial.leg(target_group.leg).joint_angles_rad[joint_index]
    delivered_trace: list[tuple[int, ...]] = []
    torque_trace: list[tuple[tuple[float, float, float], ...]] = []
    body_trace: list[dict[str, object]] = []
    contact_sets: list[frozenset[str]] = []
    support_margins = [initial.support_margin_m]
    for scheduled in schedule.spikes_by_step:
        delivered = tuple(neuron_id for neuron_id in scheduled if neuron_id not in selected)
        before = simulator.observe()
        phases = (
            before.legs[0].phase,
            before.legs[1].phase,
            before.legs[2].phase,
            before.legs[3].phase,
            before.legs[4].phase,
            before.legs[5].phase,
        )
        state = decoder.decode(
            delivered,
            window_s=parameters.dt_s,
            leg_phases=phases,
        )
        body = simulator.step(state.torques)
        delivered_trace.append(delivered)
        torque_trace.append(state.torques.values)
        body_trace.append(body.model_dump(mode="json"))
        contact_sets.append(
            frozenset(leg.name for leg in body.legs if leg.contact)
        )
        support_margins.append(body.support_margin_m)
    final = simulator.observe()
    joint_motion: float | None = None
    if target_group is not None and initial_angle is not None:
        joint_index = JOINT_NAMES.index(target_group.joint)
        joint_motion = (
            final.leg(target_group.leg).joint_angles_rad[joint_index] - initial_angle
        )
    trace_payload = {
        "spikes": delivered_trace,
        "torques": torque_trace,
        "bodies": body_trace,
    }
    return HexapodConditionResult(
        name=name,
        protocol=protocol,
        schedule_name=schedule.name,
        schedule_digest=schedule.sha256,
        spike_digest=_digest(delivered_trace),
        torque_digest=_digest(torque_trace),
        trace_digest=_digest(trace_payload),
        stimulated_population=target_group.name if target_group is not None else None,
        silenced_populations=tuple(sorted(silenced)),
        motor_spikes=sum(len(spikes) for spikes in delivered_trace),
        joint_motion_rad=joint_motion,
        forward_displacement_m=final.thorax_position_m[0] - initial.thorax_position_m[0],
        lateral_drift_m=abs(final.thorax_position_m[1] - initial.thorax_position_m[1]),
        minimum_support_margin_m=min(support_margins),
        alternating_support=_alternating_support(tuple(contact_sets)),
        energy_j=final.energy_j,
        fallen=final.fallen,
        final_body=final,
    )


def _variant_parameters(
    base: HexapodParameters,
    *,
    damping_scale: float = 1.0,
    body_mass_scale: float = 1.0,
    friction_scale: float = 1.0,
) -> HexapodParameters:
    payload = base.model_dump(mode="python")
    payload["joint_damping_nm_s_rad"] = tuple(
        value * damping_scale for value in base.joint_damping_nm_s_rad
    )
    payload["body_mass_kg"] = base.body_mass_kg * body_mass_scale
    payload["friction_coefficient"] = base.friction_coefficient * friction_scale
    return HexapodParameters.model_validate(payload)


def _mirror_group(mapping: HexapodMotorMap, group: MotorGroup) -> MotorGroup:
    side, remainder = group.name.split("_", maxsplit=1)
    mirrored_side = "right" if side == "left" else "left"
    return mapping.group(f"{mirrored_side}_{remainder}")


def _opposite_group(mapping: HexapodMotorMap, group: MotorGroup) -> MotorGroup:
    opposite = "extensor" if group.direction == "flexor" else "flexor"
    return mapping.group(f"{group.leg}_{group.joint}_{opposite}")


def _lesion_fraction(normal: float, lesion: float) -> float:
    denominator = max(abs(normal), 1e-15)
    return (abs(normal) - abs(lesion)) / denominator


def _direct_classification(
    group: MotorGroup,
    normal: HexapodConditionResult,
    lesion: HexapodConditionResult,
    restored: HexapodConditionResult,
    replay: HexapodConditionResult,
    mirror: HexapodConditionResult,
    perturbation: HexapodConditionResult,
    holdout: HexapodConditionResult,
    thresholds: HexapodBenchmarkThresholds,
) -> ClaimClassification:
    assert normal.joint_motion_rad is not None
    assert lesion.joint_motion_rad is not None
    effect = normal.joint_motion_rad
    lesion_effect = _lesion_fraction(effect, lesion.joint_motion_rad)
    expected_sign = -1.0 if group.direction == "flexor" else 1.0
    directional_values = tuple(
        item.joint_motion_rad
        for item in (normal, mirror, perturbation, holdout)
    )
    if normal.motor_spikes < thresholds.minimum_motor_spikes:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
        reasons = ("direct motor spike count is below the fixed threshold",)
    elif any(
        value is None
        or expected_sign * value < thresholds.minimum_joint_motion_rad
        for value in directional_values
    ):
        classification = "directionally_wrong"
        reasons = ("normal, mirror, perturbation, or holdout joint sign is wrong",)
    elif lesion_effect < thresholds.minimum_lesion_fraction:
        classification = "null"
        reasons = ("matching motor lesion effect is below threshold",)
    elif (
        restored.trace_digest != normal.trace_digest
        or replay.trace_digest != normal.trace_digest
    ):
        classification = "null"
        reasons = ("restoration or exact replay failed",)
    else:
        classification = "positive"
        reasons = (
            "direction, lesion, restoration, replay, perturbation, and holdout gates passed",
        )
    return ClaimClassification(
        classification=classification,
        numerator=effect,
        denominator=max(abs(effect), thresholds.minimum_joint_motion_rad),
        threshold=thresholds.minimum_joint_motion_rad,
        lesion_fraction=lesion_effect,
        reasons=reasons,
    )


def _gait_classification(
    normal: HexapodConditionResult,
    lesion: HexapodConditionResult,
    restored: HexapodConditionResult,
    replay: HexapodConditionResult,
    perturbation: HexapodConditionResult,
    holdout: HexapodConditionResult,
    thresholds: HexapodBenchmarkThresholds,
) -> ClaimClassification:
    effect = normal.forward_displacement_m
    lesion_effect = _lesion_fraction(effect, lesion.forward_displacement_m)
    if normal.motor_spikes < thresholds.minimum_motor_spikes:
        classification: Literal[
            "positive", "null", "directionally_wrong", "underpowered"
        ] = "underpowered"
        reasons = ("phase-gait motor spike count is below the fixed threshold",)
    elif effect < -thresholds.minimum_forward_displacement_m:
        classification = "directionally_wrong"
        reasons = ("phase-gait displacement is opposite the predeclared forward axis",)
    elif effect < thresholds.minimum_forward_displacement_m:
        classification = "null"
        reasons = ("forward displacement is below the fixed threshold",)
    elif (
        normal.lateral_drift_m > thresholds.maximum_lateral_drift_m
        or normal.minimum_support_margin_m < thresholds.minimum_support_margin_m
        or normal.fallen
        or not normal.alternating_support
    ):
        classification = "null"
        reasons = ("stability, lateral drift, fall, or alternating support gate failed",)
    elif lesion_effect < thresholds.minimum_lesion_fraction:
        classification = "null"
        reasons = ("bilateral motor lesion effect is below threshold",)
    elif (
        restored.trace_digest != normal.trace_digest
        or replay.trace_digest != normal.trace_digest
        or perturbation.forward_displacement_m < thresholds.minimum_forward_displacement_m
        or holdout.forward_displacement_m < thresholds.minimum_forward_displacement_m
    ):
        classification = "null"
        reasons = ("restoration, replay, perturbation, or holdout gate failed",)
    else:
        classification = "positive"
        reasons = ("all predeclared phase-gait gates passed",)
    return ClaimClassification(
        classification=classification,
        numerator=effect,
        denominator=max(abs(effect), thresholds.minimum_forward_displacement_m),
        threshold=thresholds.minimum_forward_displacement_m,
        lesion_fraction=lesion_effect,
        reasons=reasons,
    )


def run_hexapod_benchmark(
    graph: EventConnectome,
    mapping: HexapodMotorMap,
    *,
    steps: int,
    thresholds: HexapodBenchmarkThresholds,
    parameters: HexapodParameters | None = None,
) -> HexapodBenchmarkResult:
    """Run all predeclared direct-motor and phase-gait causal conditions."""

    mapping.validate_graph(graph)
    base = parameters or HexapodParameters()
    direct_schedules = build_direct_motor_schedules(mapping, steps=steps)
    gait_schedule = _build_phase_gait_schedule(mapping, base, steps=steps)
    schedules = (*direct_schedules, gait_schedule)
    schedules_by_group = {
        group.name: schedule
        for group, schedule in zip(mapping.groups, direct_schedules, strict=True)
    }
    before_digest = _graph_digest(graph)
    conditions: list[HexapodConditionResult] = []
    comparisons: list[CausalComparison] = []
    claims: dict[str, ClaimClassification] = {}
    perturbation_parameters = _variant_parameters(base, damping_scale=1.1)
    holdout_parameters = _variant_parameters(
        base,
        body_mass_scale=1.2,
        friction_scale=0.8,
    )

    for group in mapping.groups:
        schedule = schedules_by_group[group.name]
        prefix = f"direct:{group.leg}:{group.joint}:{group.direction}"
        normal = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:normal",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            target_group=group,
        )
        lesion = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:matching_lesion",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            silenced=frozenset({group.name}),
            target_group=group,
        )
        opposite_group = _opposite_group(mapping, group)
        opposite = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:opposite_lesion",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            silenced=frozenset({opposite_group.name}),
            target_group=group,
        )
        restored = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:restored",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            target_group=group,
        )
        replay = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:replay",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            target_group=group,
        )
        mirrored_group = _mirror_group(mapping, group)
        mirror = _run_condition(
            mapping,
            schedules_by_group[mirrored_group.name],
            name=f"{prefix}:mirror",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=base,
            target_group=mirrored_group,
        )
        perturbation = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:perturbation",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=perturbation_parameters,
            target_group=group,
        )
        holdout = _run_condition(
            mapping,
            schedule,
            name=f"{prefix}:holdout",
            protocol=ProtocolName.MOTOR_GROUP_CALIBRATION,
            parameters=holdout_parameters,
            target_group=group,
        )
        group_conditions = (
            normal,
            lesion,
            opposite,
            restored,
            replay,
            mirror,
            perturbation,
            holdout,
        )
        conditions.extend(group_conditions)
        assert normal.joint_motion_rad is not None
        assert lesion.joint_motion_rad is not None
        comparisons.append(
            CausalComparison(
                name=f"{group.name}:matching_lesion",
                normal_condition=normal.name,
                intervention_condition=lesion.name,
                metric="joint_motion_rad",
                normal_value=normal.joint_motion_rad,
                intervention_value=lesion.joint_motion_rad,
                lesion_fraction=_lesion_fraction(
                    normal.joint_motion_rad, lesion.joint_motion_rad
                ),
            )
        )
        claims[group.name] = _direct_classification(
            group,
            normal,
            lesion,
            restored,
            replay,
            mirror,
            perturbation,
            holdout,
            thresholds,
        )

    all_motor_names = frozenset(group.name for group in mapping.groups)
    gait_normal = _run_condition(
        mapping,
        gait_schedule,
        name="gait:normal",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=base,
    )
    gait_lesion = _run_condition(
        mapping,
        gait_schedule,
        name="gait:bilateral_lesion",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=base,
        silenced=all_motor_names,
    )
    gait_restored = _run_condition(
        mapping,
        gait_schedule,
        name="gait:restored",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=base,
    )
    gait_replay = _run_condition(
        mapping,
        gait_schedule,
        name="gait:replay",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=base,
    )
    gait_perturbation = _run_condition(
        mapping,
        gait_schedule,
        name="gait:perturbation",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=perturbation_parameters,
    )
    gait_holdout = _run_condition(
        mapping,
        gait_schedule,
        name="gait:holdout",
        protocol=ProtocolName.PHASE_GAIT_CALIBRATION,
        parameters=holdout_parameters,
    )
    gait_conditions = (
        gait_normal,
        gait_lesion,
        gait_restored,
        gait_replay,
        gait_perturbation,
        gait_holdout,
    )
    conditions.extend(gait_conditions)
    comparisons.append(
        CausalComparison(
            name="gait:bilateral_lesion",
            normal_condition=gait_normal.name,
            intervention_condition=gait_lesion.name,
            metric="forward_displacement_m",
            normal_value=gait_normal.forward_displacement_m,
            intervention_value=gait_lesion.forward_displacement_m,
            lesion_fraction=_lesion_fraction(
                gait_normal.forward_displacement_m,
                gait_lesion.forward_displacement_m,
            ),
        )
    )
    gait_claim = _gait_classification(
        gait_normal,
        gait_lesion,
        gait_restored,
        gait_replay,
        gait_perturbation,
        gait_holdout,
        thresholds,
    )

    graph_unchanged = _graph_digest(graph) == before_digest
    by_name = {condition.name: condition for condition in conditions}
    replay_pairs = []
    for group in mapping.groups:
        prefix = f"direct:{group.leg}:{group.joint}:{group.direction}"
        replay_pairs.append(
            (by_name[f"{prefix}:replay"], by_name[f"{prefix}:normal"])
        )
    replay_pairs.append((gait_replay, gait_normal))
    replay_exact = all(
        replay.trace_digest == normal.trace_digest
        for replay, normal in replay_pairs
    )
    condition_tuple = tuple(conditions)
    joint_bounds = all(
        lower <= angle <= upper
        for condition in condition_tuple
        for leg in condition.final_body.legs
        for angle, lower, upper in zip(
            leg.joint_angles_rad,
            base.joint_lower_rad,
            base.joint_upper_rad,
            strict=True,
        )
    )
    finite_state = all(
        math.isfinite(value)
        for condition in condition_tuple
        for value in (
            condition.forward_displacement_m,
            condition.lateral_drift_m,
            condition.minimum_support_margin_m,
            condition.energy_j,
        )
    )
    nonnegative_energy = all(condition.energy_j >= 0.0 for condition in condition_tuple)
    gates = {
        "direct_motor_claims": all(
            claim.classification == "positive" for claim in claims.values()
        ),
        "joint_bounds": joint_bounds,
        "finite_state": finite_state,
        "nonnegative_energy": nonnegative_energy,
        "replay_exact": replay_exact,
        "graph_unchanged": graph_unchanged,
    }
    assumptions = (
        BenchmarkAssumption(
            name="hexapod_parameters",
            value_sha256=_digest(base.model_dump(mode="json")),
        ),
        BenchmarkAssumption(
            name="classification_thresholds",
            value_sha256=_digest(thresholds.model_dump(mode="json")),
        ),
        BenchmarkAssumption(
            name="target_independent_phase_gait",
            value_sha256=_digest(gait_schedule.model_dump(mode="json")),
        ),
    )
    return HexapodBenchmarkResult(
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        thresholds=thresholds,
        assumptions=assumptions,
        schedules=schedules,
        conditions=condition_tuple,
        comparisons=tuple(comparisons),
        direct_motor_claims=claims,
        gait_claim=gait_claim,
        calibration_gates=gates,
        calibration_passed=all(gates.values()),
        replay_exact=replay_exact,
        graph_unchanged=graph_unchanged,
    )
