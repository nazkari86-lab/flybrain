"""Signed MBON-to-descending anatomy and fail-closed causal calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flybrain.biological_registry import (
    load_biological_registry,
    resolve_biological_registry,
)
from flybrain.descending_interface import DescendingMap
from flybrain.graph import EventConnectome, SparseConnectome
from flybrain.hexapod_motor import HexapodMotorDecoder, HexapodMotorMap
from flybrain.mushroom_body_learning import derive_approach_mbon_ids
from flybrain.plastic_edge_registry import resolve_plastic_edge_manifests
from flybrain.provenance import snapshot_content_sha256
from flybrain.reinforcement_interface import ReinforcementInterface
from flybrain.shiu import ShiuParameters, simulate_shiu

PathSign = Literal[-1, 0, 1]
AuthorityClass = Literal[
    "excitatory_direct",
    "inhibitory_direct",
    "sign_zero_direct",
    "mixed_direct",
    "multihop_only",
    "disconnected",
]


class DirectDescendingEdge(BaseModel, frozen=True):
    """One retained MBON edge into a registered descending population."""

    model_config = ConfigDict(extra="forbid")

    target_id: int = Field(gt=0)
    target_population: str = Field(min_length=1)
    sign: PathSign
    weight_magnitude: float = Field(ge=0.0)


class MbonPathwayProfile(BaseModel, frozen=True):
    """Shortest signed reachability for one learned MBON."""

    model_config = ConfigDict(extra="forbid")

    mbon_id: int = Field(gt=0)
    learned_role: Literal["approach", "avoidance"]
    transmitter: str
    outgoing_edge_count: int = Field(ge=0)
    outgoing_weight_magnitude: float = Field(ge=0.0)
    direct_selected_dn_edges: tuple[DirectDescendingEdge, ...]
    selected_dn_shortest_hops: int | None = Field(default=None, ge=1)
    selected_dn_shortest_signs: tuple[PathSign, ...] = ()
    motor_shortest_hops: int | None = Field(default=None, ge=1)
    motor_shortest_signs: tuple[PathSign, ...] = ()
    authority_class: AuthorityClass


class MatchedMbonControl(BaseModel, frozen=True):
    """Deterministic transmitter- and output-degree-matched control pairing."""

    model_config = ConfigDict(extra="forbid")

    source_mbon_id: int = Field(gt=0)
    control_mbon_id: int = Field(gt=0)
    transmitter_matched: bool


class MbonDescendingPathwayAudit(BaseModel, frozen=True):
    """Dataset-derived route audit; no functional claim is inferred from anatomy."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["mbon-descending-pathway-audit-v1"] = (
        "mbon-descending-pathway-audit-v1"
    )
    evidence_kind: Literal["dataset_measurement"] = "dataset_measurement"
    max_hops: int = Field(ge=1)
    learned_mbon_count: int = Field(gt=0)
    approach_mbon_count: int = Field(ge=0)
    avoidance_mbon_count: int = Field(ge=0)
    direct_selected_dn_edge_count: int = Field(ge=0)
    direct_selected_dn_sign_counts: dict[int, int]
    excitatory_direct_mbon_ids: tuple[int, ...]
    matched_control_mbon_ids: tuple[int, ...]
    matched_controls: tuple[MatchedMbonControl, ...]
    profiles: tuple[MbonPathwayProfile, ...]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def profile(self, mbon_id: int) -> MbonPathwayProfile:
        matches = tuple(item for item in self.profiles if item.mbon_id == mbon_id)
        if len(matches) != 1:
            raise ValueError(f"MBON profile not found exactly once: {mbon_id}")
        return matches[0]


class MbonDescendingCondition(BaseModel, frozen=True):
    """One fixed-source condition measured at DN and motor populations."""

    model_config = ConfigDict(extra="forbid")

    name: str
    stimulated_mbon_ids: tuple[int, ...]
    silenced_ids: tuple[int, ...]
    source_voltage_events: int = Field(ge=0)
    source_spikes: int = Field(ge=0)
    descending_spikes: dict[str, int]
    selected_descending_spikes: int = Field(ge=0)
    motor_spikes: int = Field(ge=0)
    decoded_torque_l1_nm_s: float = Field(ge=0.0)
    active_motor_groups: int = Field(ge=0)
    total_spikes: int = Field(ge=0)
    trace_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class MbonDescendingCausalAssayResult(BaseModel, frozen=True):
    """Causal pathway evidence separated from motor-specific and behavioral claims."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["mbon-descending-causal-assay-v1"] = (
        "mbon-descending-causal-assay-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    conditions: dict[str, MbonDescendingCondition]
    gates: dict[str, bool]
    causal_mbon_to_descending_claim_allowed: bool
    specific_mbon_dn_motor_claim_allowed: bool
    autonomous_behavior_claim_allowed: Literal[False] = False
    replay_exact: bool
    restoration_exact: bool
    graph_unchanged: bool


class RetainedMbonDescendingAssayResult(BaseModel, frozen=True):
    """One exact MaleCNS route audit plus causal calibration."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["retained-mbon-descending-assay-v1"] = (
        "retained-mbon-descending-assay-v1"
    )
    evidence_kind: Literal["simulation_observation"] = "simulation_observation"
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    learning_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    motor_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_neurons: int = Field(gt=0)
    graph_edges: int = Field(gt=0)
    pathways: MbonDescendingPathwayAudit
    causal: MbonDescendingCausalAssayResult
    autonomous_behavior_claim_allowed: Literal[False] = False


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


def _validated_ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integer IDs")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must contain sorted unique IDs")
    return values


def _path_signs(bits: int) -> tuple[PathSign, ...]:
    values: list[PathSign] = []
    if bits & 2:
        values.append(-1)
    if bits & 4:
        values.append(0)
    if bits & 1:
        values.append(1)
    return tuple(values)


def _reverse_shortest_reachability(
    graph: EventConnectome,
    *,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    max_hops: int,
) -> dict[int, tuple[int, tuple[PathSign, ...]]]:
    incoming = graph.outgoing.transpose().tocsr()
    current = np.zeros(graph.neuron_count, dtype=np.uint8)
    current[target_indices] = 1
    seen = np.zeros(graph.neuron_count, dtype=np.bool_)
    seen[target_indices] = True
    result: dict[int, tuple[int, tuple[PathSign, ...]]] = {}
    source_set = set(int(value) for value in source_indices)
    for depth in range(1, max_hops + 1):
        next_state = np.zeros_like(current)
        for post_index in np.flatnonzero(current):
            start = incoming.indptr[post_index]
            stop = incoming.indptr[post_index + 1]
            presynaptic = incoming.indices[start:stop]
            edge_signs = np.sign(incoming.data[start:stop]).astype(np.int8, copy=False)
            state = int(current[post_index])
            for bit, positive, negative in ((1, 1, 2), (2, 2, 1), (4, 4, 4)):
                if state & bit:
                    np.bitwise_or.at(next_state, presynaptic[edge_signs > 0], positive)
                    np.bitwise_or.at(next_state, presynaptic[edge_signs < 0], negative)
                    np.bitwise_or.at(next_state, presynaptic[edge_signs == 0], 4)
        next_state[seen] = 0
        for source_index in source_set:
            bits = int(next_state[source_index])
            if bits and source_index not in result:
                result[source_index] = (depth, _path_signs(bits))
        seen |= next_state.astype(np.bool_)
        current = next_state
        if not np.any(current):
            break
    return result


def _authority_class(
    direct_edges: tuple[DirectDescendingEdge, ...],
    shortest_hops: int | None,
) -> AuthorityClass:
    signs = {edge.sign for edge in direct_edges}
    if signs == {1}:
        return "excitatory_direct"
    if signs == {-1}:
        return "inhibitory_direct"
    if signs == {0}:
        return "sign_zero_direct"
    if signs:
        return "mixed_direct"
    if shortest_hops is not None:
        return "multihop_only"
    return "disconnected"


def _matched_controls(
    profiles: tuple[MbonPathwayProfile, ...],
    source_ids: tuple[int, ...],
) -> tuple[MatchedMbonControl, ...]:
    by_id = {item.mbon_id: item for item in profiles}
    used: set[int] = set()
    result: list[MatchedMbonControl] = []
    for source_id in source_ids:
        source = by_id[source_id]
        candidates = [
            item
            for item in profiles
            if item.mbon_id not in source_ids
            and item.mbon_id not in used
            and not item.direct_selected_dn_edges
        ]
        same_transmitter = [
            item for item in candidates if item.transmitter == source.transmitter
        ]
        pool = same_transmitter or candidates
        if not pool:
            raise ValueError("not enough non-direct MBONs for matched controls")

        def distance(
            item: MbonPathwayProfile,
            source_profile: MbonPathwayProfile = source,
        ) -> tuple[float, int]:
            value = abs(
                math.log1p(item.outgoing_edge_count)
                - math.log1p(source_profile.outgoing_edge_count)
            ) + abs(
                math.log1p(item.outgoing_weight_magnitude)
                - math.log1p(source_profile.outgoing_weight_magnitude)
            )
            return value, item.mbon_id

        selected = min(pool, key=distance)
        used.add(selected.mbon_id)
        result.append(
            MatchedMbonControl(
                source_mbon_id=source_id,
                control_mbon_id=selected.mbon_id,
                transmitter_matched=selected.transmitter == source.transmitter,
            )
        )
    return tuple(result)


def audit_mbon_descending_pathways(
    graph: EventConnectome,
    *,
    mbon_ids: tuple[int, ...],
    approach_mbon_ids: tuple[int, ...],
    descending: DescendingMap,
    motor: HexapodMotorMap,
    max_hops: int = 4,
) -> MbonDescendingPathwayAudit:
    """Measure direct edges and shortest signed paths without inventing mappings."""

    mbon_ids = _validated_ids(mbon_ids, "MBON IDs")
    if type(max_hops) is not int or max_hops < 1:
        raise ValueError("max_hops must be a positive integer")
    available = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    required = {
        *mbon_ids,
        *(value for values in descending.named_populations().values() for value in values),
        *(value for group in motor.groups for value in group.neuron_ids),
    }
    missing = sorted(required - available.keys())
    if missing:
        raise ValueError(f"pathway audit IDs absent from graph: {missing}")
    approach = set(approach_mbon_ids) & set(mbon_ids)
    population_by_id = {
        neuron_id: name
        for name, values in descending.named_populations().items()
        for neuron_id in values
    }
    descending_indices = np.asarray(
        [available[value] for value in population_by_id], dtype=np.int64
    )
    motor_indices = np.asarray(
        [available[value] for group in motor.groups for value in group.neuron_ids],
        dtype=np.int64,
    )
    mbon_indices = np.asarray([available[value] for value in mbon_ids], dtype=np.int64)
    descending_reach = _reverse_shortest_reachability(
        graph,
        source_indices=mbon_indices,
        target_indices=descending_indices,
        max_hops=max_hops,
    )
    motor_reach = _reverse_shortest_reachability(
        graph,
        source_indices=mbon_indices,
        target_indices=motor_indices,
        max_hops=max_hops,
    )
    descending_index_set = set(int(value) for value in descending_indices)
    profiles: list[MbonPathwayProfile] = []
    all_direct_edges: list[DirectDescendingEdge] = []
    for mbon_id, source_index in zip(mbon_ids, mbon_indices, strict=True):
        start = graph.outgoing.indptr[source_index]
        stop = graph.outgoing.indptr[source_index + 1]
        posts = graph.outgoing.indices[start:stop]
        weights = graph.outgoing.data[start:stop]
        direct_edges = tuple(
            sorted(
                (
                    DirectDescendingEdge(
                        target_id=int(graph.neuron_ids[post_index]),
                        target_population=population_by_id[int(graph.neuron_ids[post_index])],
                        sign=cast(PathSign, int(np.sign(weight))),
                        weight_magnitude=float(abs(weight)),
                    )
                    for post_index, weight in zip(posts, weights, strict=True)
                    if int(post_index) in descending_index_set
                ),
                key=lambda item: (item.target_population, item.target_id),
            )
        )
        all_direct_edges.extend(direct_edges)
        dn_path = descending_reach.get(int(source_index))
        motor_path = motor_reach.get(int(source_index))
        dn_hops = dn_path[0] if dn_path is not None else None
        profiles.append(
            MbonPathwayProfile(
                mbon_id=mbon_id,
                learned_role="approach" if mbon_id in approach else "avoidance",
                transmitter=(
                    graph.transmitters[source_index]
                    if len(graph.transmitters) == graph.neuron_count
                    else "unresolved"
                ),
                outgoing_edge_count=int(stop - start),
                outgoing_weight_magnitude=float(np.abs(weights).sum()),
                direct_selected_dn_edges=direct_edges,
                selected_dn_shortest_hops=dn_hops,
                selected_dn_shortest_signs=dn_path[1] if dn_path is not None else (),
                motor_shortest_hops=motor_path[0] if motor_path is not None else None,
                motor_shortest_signs=motor_path[1] if motor_path is not None else (),
                authority_class=_authority_class(direct_edges, dn_hops),
            )
        )
    resolved_profiles = tuple(profiles)
    source_ids = tuple(
        item.mbon_id
        for item in resolved_profiles
        if item.authority_class == "excitatory_direct"
    )
    if not source_ids:
        raise ValueError("no excitatory direct MBONs reach registered descending populations")
    controls = _matched_controls(resolved_profiles, source_ids)
    sign_counts = {
        sign: sum(edge.sign == sign for edge in all_direct_edges)
        for sign in (-1, 0, 1)
    }
    payload = {
        "max_hops": max_hops,
        "profiles": [item.model_dump(mode="json") for item in resolved_profiles],
        "controls": [item.model_dump(mode="json") for item in controls],
    }
    return MbonDescendingPathwayAudit(
        max_hops=max_hops,
        learned_mbon_count=len(mbon_ids),
        approach_mbon_count=len(approach),
        avoidance_mbon_count=len(mbon_ids) - len(approach),
        direct_selected_dn_edge_count=len(all_direct_edges),
        direct_selected_dn_sign_counts=sign_counts,
        excitatory_direct_mbon_ids=source_ids,
        matched_control_mbon_ids=tuple(item.control_mbon_id for item in controls),
        matched_controls=controls,
        profiles=resolved_profiles,
        audit_sha256=hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    )


def _run_condition(
    graph: EventConnectome,
    *,
    name: str,
    stimulated_mbon_ids: tuple[int, ...],
    silenced_ids: tuple[int, ...],
    descending: DescendingMap,
    motor: HexapodMotorMap,
    steps: int,
    seed: int,
    parameters: ShiuParameters,
) -> MbonDescendingCondition:
    index_by_id = {int(value): index for index, value in enumerate(graph.neuron_ids)}
    source_indices = np.asarray(
        [index_by_id[value] for value in stimulated_mbon_ids], dtype=np.int64
    )
    amplitude = np.float32(parameters.synapse_mv * parameters.poisson_voltage_scale)
    event_steps = range(0, steps, parameters.refractory_steps + 1)
    events = {
        step: (
            source_indices,
            np.full(source_indices.size, amplitude, dtype=np.float32),
        )
        for step in event_steps
        if source_indices.size
    }
    silence_mask = np.isin(graph.neuron_ids, np.asarray(silenced_ids, dtype=np.uint64))
    descending_populations = descending.named_populations()
    descending_counts = dict.fromkeys(descending_populations, 0)
    motor_owner = {
        neuron_id: group.name for group in motor.groups for neuron_id in group.neuron_ids
    }
    source_set = set(stimulated_mbon_ids)
    source_spikes = 0
    motor_spikes = 0
    total_spikes = 0
    active_motor_groups: set[str] = set()
    motor_decoder = HexapodMotorDecoder(motor, descending_map=descending)
    motor_and_descending_ids = set(motor_owner) | {
        value for ids in descending_populations.values() for value in ids
    }
    window_spikes: list[int] = []
    decoded_torque_l1_nm_s = 0.0
    window_steps = 10
    digest = hashlib.sha256()
    for step_index, batch in enumerate(simulate_shiu(
        graph,
        parameters,
        steps=steps,
        external_voltage_events=events,
        seed=seed,
        silenced=silence_mask,
    )):
        fired = tuple(int(value) for value in batch.neuron_ids)
        digest.update(f"{batch.step}:".encode())
        digest.update(np.asarray(fired, dtype=np.uint64).tobytes())
        total_spikes += len(fired)
        source_spikes += sum(value in source_set for value in fired)
        for population, ids in descending_populations.items():
            descending_counts[population] += sum(value in ids for value in fired)
        for value in fired:
            group = motor_owner.get(value)
            if group is not None:
                motor_spikes += 1
                active_motor_groups.add(group)
            if value in motor_and_descending_ids:
                window_spikes.append(value)
        elapsed_steps = step_index % window_steps + 1
        if elapsed_steps == window_steps or step_index == steps - 1:
            duration_s = elapsed_steps * parameters.dt_ms / 1000.0
            state = motor_decoder.decode(
                tuple(window_spikes),
                window_s=duration_s,
                leg_phases=(0.0,) * 6,
            )
            decoded_torque_l1_nm_s += duration_s * sum(
                abs(value) for leg in state.torques.values for value in leg
            )
            window_spikes.clear()
    return MbonDescendingCondition(
        name=name,
        stimulated_mbon_ids=stimulated_mbon_ids,
        silenced_ids=silenced_ids,
        source_voltage_events=len(events) * len(stimulated_mbon_ids),
        source_spikes=source_spikes,
        descending_spikes=descending_counts,
        selected_descending_spikes=sum(descending_counts.values()),
        motor_spikes=motor_spikes,
        decoded_torque_l1_nm_s=decoded_torque_l1_nm_s,
        active_motor_groups=len(active_motor_groups),
        total_spikes=total_spikes,
        trace_digest=digest.hexdigest(),
    )


def run_mbon_descending_causal_assay(
    graph: EventConnectome,
    *,
    audit: MbonDescendingPathwayAudit,
    descending: DescendingMap,
    motor: HexapodMotorMap,
    steps: int = 500,
    seed: int = 7,
    parameters: ShiuParameters | None = None,
) -> MbonDescendingCausalAssayResult:
    """Run stimulation, lesions, restoration, replay, and a matched control."""

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    active_parameters = parameters or ShiuParameters(
        dt_ms=0.1,
        refractory_ms=2.0,
        synaptic_delay_ms=1.0,
    )
    active_parameters.validate()
    motor.validate_graph(graph)
    before = _graph_digest(graph)
    sources = audit.excitatory_direct_mbon_ids
    controls = audit.matched_control_mbon_ids
    descending_ids = tuple(
        sorted(
            value
            for values in descending.named_populations().values()
            for value in values
        )
    )

    def run(
        name: str,
        stimulated: tuple[int, ...],
        silenced: tuple[int, ...] = (),
    ) -> MbonDescendingCondition:
        return _run_condition(
            graph,
            name=name,
            stimulated_mbon_ids=stimulated,
            silenced_ids=silenced,
            descending=descending,
            motor=motor,
            steps=steps,
            seed=seed,
            parameters=active_parameters,
        )

    conditions = {
        "static": run("static", ()),
        "normal": run("normal", sources),
        "source_lesion": run("source_lesion", sources, sources),
        "descending_lesion": run("descending_lesion", sources, descending_ids),
        "restored": run("restored", sources),
        "replay": run("replay", sources),
        "matched_control": run("matched_control", controls),
    }
    static = conditions["static"]
    normal = conditions["normal"]
    source_lesion = conditions["source_lesion"]
    descending_lesion = conditions["descending_lesion"]
    matched = conditions["matched_control"]
    restoration_exact = conditions["restored"].trace_digest == normal.trace_digest
    replay_exact = conditions["replay"].trace_digest == normal.trace_digest
    graph_unchanged = _graph_digest(graph) == before
    gates = {
        "source_recruited": normal.source_spikes > 0,
        "selected_descending_recruited": (
            normal.selected_descending_spikes > static.selected_descending_spikes
        ),
        "motor_recruited": normal.motor_spikes > static.motor_spikes,
        "source_lesion_blocks_descending": (
            source_lesion.selected_descending_spikes
            == static.selected_descending_spikes
        ),
        "source_lesion_blocks_motor": source_lesion.motor_spikes == static.motor_spikes,
        "descending_lesion_reduces_motor": (
            descending_lesion.motor_spikes < normal.motor_spikes
        ),
        "matched_control_has_less_motor": matched.motor_spikes < normal.motor_spikes,
        "restoration_exact": restoration_exact,
        "replay_exact": replay_exact,
        "graph_unchanged": graph_unchanged,
    }
    causal = all(
        gates[name]
        for name in (
            "source_recruited",
            "selected_descending_recruited",
            "source_lesion_blocks_descending",
            "restoration_exact",
            "replay_exact",
            "graph_unchanged",
        )
    )
    specific_motor = causal and all(
        gates[name]
        for name in (
            "motor_recruited",
            "source_lesion_blocks_motor",
            "descending_lesion_reduces_motor",
            "matched_control_has_less_motor",
        )
    )
    return MbonDescendingCausalAssayResult(
        steps=steps,
        seed=seed,
        conditions=conditions,
        gates=gates,
        causal_mbon_to_descending_claim_allowed=causal,
        specific_mbon_dn_motor_claim_allowed=specific_motor,
        replay_exact=replay_exact,
        restoration_exact=restoration_exact,
        graph_unchanged=graph_unchanged,
    )


def run_retained_mbon_descending_assay(
    snapshot: Path,
    *,
    learning_registry_path: Path = Path(
        "data/registry/autonomous-learning-registry-v1.json"
    ),
    motor_registry_path: Path = Path("data/registry/hexapod-motor-registry-v2.json"),
    steps: int = 500,
    seed: int = 7,
    max_hops: int = 4,
) -> RetainedMbonDescendingAssayResult:
    """Resolve and test the retained learned-MBON route without synthetic edges."""

    learning_registry = load_biological_registry(learning_registry_path)
    motor_registry = load_biological_registry(motor_registry_path)
    learning_populations = resolve_biological_registry(learning_registry, snapshot)
    motor_populations = resolve_biological_registry(motor_registry, snapshot)
    manifests = {
        manifest.name: manifest
        for manifest in resolve_plastic_edge_manifests(
            learning_registry,
            learning_populations,
            snapshot,
        )
    }
    kc_to_mbon = manifests["kc_to_mbon"]
    dan_to_mbon = manifests["dan_to_mbon"]
    reinforcement = ReinforcementInterface.from_resolved_registry(
        learning_populations
    )
    approach_ids = derive_approach_mbon_ids(
        dan_ids=np.asarray(
            [pre_id for pre_id, _ in dan_to_mbon.edge_pairs], dtype=np.uint64
        ),
        dan_post_ids=np.asarray(
            [post_id for _, post_id in dan_to_mbon.edge_pairs], dtype=np.uint64
        ),
        appetitive_dan_ids=np.asarray(
            reinforcement.appetitive_dan_ids, dtype=np.uint64
        ),
        aversive_dan_ids=np.asarray(
            reinforcement.aversive_dan_ids, dtype=np.uint64
        ),
    )
    graph = EventConnectome.from_sparse(SparseConnectome.from_snapshot(snapshot))
    motor = HexapodMotorMap.from_registry(motor_populations)
    descending = DescendingMap(
        d_na02_left=motor_populations.population("d_na02_left").neuron_ids,
        d_na02_right=motor_populations.population("d_na02_right").neuron_ids,
        d_ng13_left=motor_populations.population("d_ng13_left").neuron_ids,
        d_ng13_right=motor_populations.population("d_ng13_right").neuron_ids,
        mdn_left=motor_populations.population("mdn_left").neuron_ids,
        mdn_right=motor_populations.population("mdn_right").neuron_ids,
    )
    pathways = audit_mbon_descending_pathways(
        graph,
        mbon_ids=tuple(sorted({post_id for _, post_id in kc_to_mbon.edge_pairs})),
        approach_mbon_ids=tuple(int(value) for value in approach_ids),
        descending=descending,
        motor=motor,
        max_hops=max_hops,
    )
    causal = run_mbon_descending_causal_assay(
        graph,
        audit=pathways,
        descending=descending,
        motor=motor,
        steps=steps,
        seed=seed,
    )
    return RetainedMbonDescendingAssayResult(
        snapshot_content_sha256=snapshot_content_sha256(snapshot),
        learning_registry_sha256=learning_populations.registry_sha256,
        motor_registry_sha256=motor_populations.registry_sha256,
        graph_neurons=graph.neuron_count,
        graph_edges=graph.edge_count,
        pathways=pathways,
        causal=causal,
    )
