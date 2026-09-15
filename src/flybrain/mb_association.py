"""Deterministic associative-memory benchmark on measured mushroom-body anatomy."""

from __future__ import annotations

import hashlib
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from flybrain.mushroom_body import extract_kc_mbon_edges
from flybrain.plasticity import (
    PlasticEdgeSet,
    PlasticityParameters,
    load_plastic_state,
    save_plastic_state,
)


class MBAssociationResult(BaseModel, frozen=True):
    """Auditable measurements from one cue-specific plasticity benchmark."""

    benchmark: str
    snapshot: str
    dataset_id: str
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    kenyon_cells: int
    dopamine_neurons: int
    mbons: int
    plastic_edges: int
    total_synapse_weight: int
    target_mbon_id: int
    target_connected_kcs: int
    cue_size: int
    cue_a_ids: tuple[int, ...]
    cue_b_ids: tuple[int, ...]
    trials: int
    dopamine: float
    parameters: dict[str, float]
    trained_before: float
    trained_after: float
    trained_relative_decrease: float
    untrained_before: float
    untrained_after: float
    untrained_relative_drift: float
    no_dopamine_relative_drift: float
    cleared_eligibility_relative_drift: float
    persistence_replay_exact: bool
    state_path: str
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_revision: str
    runtime_seconds: float
    peak_rss_bytes: int


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _software_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fresh_overlay(source: PlasticEdgeSet) -> PlasticEdgeSet:
    return PlasticEdgeSet.create(
        pre_ids=source.pre_ids,
        post_ids=source.post_ids,
        baseline_weights=source.baseline_weights,
    )


def _response(edges: PlasticEdgeSet, cue_ids: np.ndarray, target_mbon_id: int) -> float:
    selected = (edges.post_ids == target_mbon_id) & np.isin(edges.pre_ids, cue_ids)
    return float(np.sum(edges.effective_weights[selected], dtype=np.float64))


def _relative_drift(before: float, after: float) -> float:
    return abs(after - before) / before


def run_mb_association(
    snapshot: Path,
    *,
    state_path: Path,
    seed: int = 7,
    cue_size: int = 64,
    trials: int = 3,
    dopamine: float = 1.0,
    params: PlasticityParameters | None = None,
) -> MBAssociationResult:
    """Train one cue and measure specificity, controls, and persistent replay."""

    if cue_size <= 0:
        raise ValueError("cue_size must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if dopamine <= 0:
        raise ValueError("dopamine must be positive")
    if state_path.exists():
        raise FileExistsError(f"plastic state already exists: {state_path}")

    started = time.perf_counter()
    active_params = params or PlasticityParameters()
    active_params.validate()
    edges, anatomy = extract_kc_mbon_edges(snapshot)

    target_ids, input_counts = np.unique(edges.post_ids, return_counts=True)
    if target_ids.size == 0:
        raise ValueError("snapshot has no measured KC-to-MBON edges")
    target_mbon_id = int(target_ids[int(np.argmax(input_counts))])
    connected_kcs = np.unique(edges.pre_ids[edges.post_ids == target_mbon_id])
    if connected_kcs.size < cue_size * 2:
        raise ValueError(
            f"target MBON {target_mbon_id} has {connected_kcs.size} KC inputs; "
            f"need at least {cue_size * 2}"
        )

    shuffled = np.random.default_rng(seed).permutation(connected_kcs)
    cue_a = shuffled[:cue_size].astype(np.uint64, copy=False)
    cue_b = shuffled[cue_size : cue_size * 2].astype(np.uint64, copy=False)
    target = np.asarray([target_mbon_id], dtype=np.uint64)

    trained_before = _response(edges, cue_a, target_mbon_id)
    untrained_before = _response(edges, cue_b, target_mbon_id)
    for _ in range(trials):
        edges.clear_eligibility()
        edges.update_eligibility(
            active_pre_ids=cue_a,
            gated_post_ids=target,
            dt_ms=0.0,
            params=active_params,
        )
        edges.apply_dopamine({target_mbon_id: dopamine}, active_params)
    edges.clear_eligibility()
    trained_after = _response(edges, cue_a, target_mbon_id)
    untrained_after = _response(edges, cue_b, target_mbon_id)

    no_dopamine = _fresh_overlay(edges)
    no_dopamine_before = _response(no_dopamine, cue_a, target_mbon_id)
    no_dopamine.update_eligibility(
        active_pre_ids=cue_a,
        gated_post_ids=target,
        dt_ms=0.0,
        params=active_params,
    )
    no_dopamine.apply_dopamine({}, active_params)
    no_dopamine_after = _response(no_dopamine, cue_a, target_mbon_id)

    cleared = _fresh_overlay(edges)
    cleared_before = _response(cleared, cue_a, target_mbon_id)
    cleared.update_eligibility(
        active_pre_ids=cue_a,
        gated_post_ids=target,
        dt_ms=0.0,
        params=active_params,
    )
    cleared.clear_eligibility()
    cleared.apply_dopamine({target_mbon_id: dopamine}, active_params)
    cleared_after = _response(cleared, cue_a, target_mbon_id)

    save_plastic_state(state_path, edges, active_params)
    restored, restored_params = load_plastic_state(state_path)
    replay_exact = restored_params == active_params and np.array_equal(
        restored.effective_weights, edges.effective_weights
    )
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()

    return MBAssociationResult(
        benchmark="mb-association-v1",
        snapshot=str(snapshot.resolve()),
        dataset_id=anatomy.dataset_id,
        source_manifest_sha256=anatomy.source_manifest_sha256,
        seed=seed,
        kenyon_cells=anatomy.kenyon_cells,
        dopamine_neurons=anatomy.dopamine_neurons,
        mbons=anatomy.mbons,
        plastic_edges=anatomy.plastic_edges,
        total_synapse_weight=anatomy.total_synapse_weight,
        target_mbon_id=target_mbon_id,
        target_connected_kcs=int(connected_kcs.size),
        cue_size=cue_size,
        cue_a_ids=tuple(int(value) for value in cue_a),
        cue_b_ids=tuple(int(value) for value in cue_b),
        trials=trials,
        dopamine=dopamine,
        parameters={key: float(value) for key, value in asdict(active_params).items()},
        trained_before=trained_before,
        trained_after=trained_after,
        trained_relative_decrease=(trained_before - trained_after) / trained_before,
        untrained_before=untrained_before,
        untrained_after=untrained_after,
        untrained_relative_drift=_relative_drift(untrained_before, untrained_after),
        no_dopamine_relative_drift=_relative_drift(no_dopamine_before, no_dopamine_after),
        cleared_eligibility_relative_drift=_relative_drift(cleared_before, cleared_after),
        persistence_replay_exact=bool(replay_exact),
        state_path=str(state_path.resolve()),
        state_sha256=state_sha256,
        software_revision=_software_revision(),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
