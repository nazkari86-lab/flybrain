"""One evidence-bound retained-MaleCNS associative motor assay."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from flybrain.associative_motor_loop import (
    AssociativeMotorCalibrationResult,
    run_associative_motor_calibration,
)
from flybrain.associative_motor_pathway import (
    MbonMotorPathwayAudit,
    resolve_mbon_motor_pathway,
)
from flybrain.autonomous_learning_benchmark import AssociativeCalibrationConfig
from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.conditioning_world import ConditioningEvent, ConditioningSchedule
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_body import HexapodParameters
from flybrain.hexapod_motor import HexapodMotorMap
from flybrain.plastic_edge_binding import bind_manifest_to_graph
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.proprioceptive_interface import ProprioceptiveMap
from flybrain.provenance import snapshot_content_sha256
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters

LEARNING_REGISTRY = Path("data/registry/autonomous-learning-registry-v1.json")
MOTOR_REGISTRY = Path("data/registry/hexapod-motor-registry-v1.json")


class RetainedAssociativeMotorAssayResult(BaseModel, frozen=True):
    """All output remains simulation evidence, never a behavioral-performance claim."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["retained-associative-motor-assay-v1"] = (
        "retained-associative-motor-assay-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    autonomous_behavior_claim_allowed: Literal[False] = False
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    anatomy: MbonMotorPathwayAudit
    paired_training: AssociativeMotorCalibrationResult
    motor_lesion: AssociativeMotorCalibrationResult
    no_contact: AssociativeMotorCalibrationResult
    no_contact_preserves_overlay: bool
    paired_associative_change_detected: bool


def _no_contact_schedule(*, seed: int, body_steps: int) -> ConditioningSchedule:
    return ConditioningSchedule(
        seed=seed,
        events=tuple(
            ConditioningEvent(
                step=step,
                odor_intensity=1.0 if step == 0 else 0.0,
                appetitive_contact_intensity=0.0,
                aversive_contact_intensity=0.0,
            )
            for step in range(body_steps)
        ),
    )


def run_retained_associative_motor_assay(
    snapshot: Path,
    *,
    body_steps: int = 1,
    seed: int = 7,
) -> RetainedAssociativeMotorAssayResult:
    """Run paired, motor-lesion, and no-contact controls on one exact snapshot."""

    if type(body_steps) is not int or body_steps <= 0:
        raise ValueError("body_steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    learning_registry = load_biological_registry(LEARNING_REGISTRY)
    learning_populations = resolve_biological_registry(learning_registry, snapshot)
    motor_populations = resolve_biological_registry(
        load_biological_registry(MOTOR_REGISTRY), snapshot
    )
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(
            learning_registry,
            learning_populations,
            snapshot,
        )
    }
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    binding = bind_manifest_to_graph(graph, manifests["kc_to_mbon"])
    reinforcement = ReinforcementInterface.from_resolved_registry(learning_populations)
    motor = HexapodMotorMap.from_registry(motor_populations)
    proprio = ProprioceptiveMap.from_registry(motor_populations)
    learning = AssociativeCalibrationConfig.from_manifest(
        binding,
        manifests["dan_to_mbon"],
        reinforcement,
        valence="appetitive",
        sensory_input_ids=learning_populations.population("olfactory_sensory").neuron_ids,
        neural_chunk_steps=100,
        shiu_parameters=ShiuParameters(
            dt_ms=0.1,
            refractory_ms=2.0,
            synaptic_delay_ms=1.0,
        ),
    )
    paired_schedule = ConditioningSchedule.create(
        seed=seed,
        steps=body_steps,
        appetitive_pair_steps=(0,),
    )
    body_parameters = HexapodParameters(dt_s=0.01)

    def run(
        schedule: ConditioningSchedule,
        *,
        motor_silenced: bool = False,
    ) -> AssociativeMotorCalibrationResult:
        return run_associative_motor_calibration(
            graph,
            binding,
            learning=learning,
            motor=motor,
            proprio=proprio,
            schedule=schedule,
            reinforcement=reinforcement,
            body_parameters=body_parameters,
            motor_silenced=motor_silenced,
        )

    paired_training = run(paired_schedule)
    motor_lesion = run(paired_schedule, motor_silenced=True)
    no_contact = run(_no_contact_schedule(seed=seed, body_steps=body_steps))
    no_contact_preserves_overlay = all(
        value == 1.0 for value in no_contact.final_multipliers
    )
    return RetainedAssociativeMotorAssayResult(
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        anatomy=resolve_mbon_motor_pathway(
            snapshot,
            mbon_ids=learning_populations.population("mbons").neuron_ids,
            motor=motor,
        ),
        paired_training=paired_training,
        motor_lesion=motor_lesion,
        no_contact=no_contact,
        no_contact_preserves_overlay=no_contact_preserves_overlay,
        paired_associative_change_detected=(
            paired_training.final_multipliers != no_contact.final_multipliers
        ),
    )
