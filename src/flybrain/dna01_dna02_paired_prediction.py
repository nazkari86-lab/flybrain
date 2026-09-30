"""Locked same-fly DNa02-over-DNa01 steering prediction assay."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.io as sio
from scipy.ndimage import binary_dilation, gaussian_filter1d

from flybrain.dna02_ephys_prediction import _digest, _fit_predict, _mse
from flybrain.mb_association import _software_revision

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
PROTOCOL = "dna01-dna02-paired-real-fly-prediction-v1"
DESIGN_REVISION = "0e62d10"
SECONDARY_CODE_REVISION = "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
PRIMARY_CODE_REVISION = "55e30c19b1a18f601df1803295f0c401aec3c167"
FILES = {
    "01": (11634626, "9cfaf03120f6ae7f09f4766b8dfef3c9"),
    "02": (11634627, "eea3f8e609abf138c4f223d56162a196"),
    "03": (11634628, "0b570dce1281528e1ad032b5ec7dfda3"),
    "04": (11634629, "1e6301518a7c15954c26ec2743df58ec"),
}
LAG_BINS = 3
GAP_BINS = 100
NULL_DRAWS = 999
NULL_SEED = 20260930


def _channel_keys(name: str) -> tuple[str, str]:
    """Return (DNa02, DNa01) raw channels, including the documented fly-03 swap."""
    if name not in FILES:
        raise ValueError(f"unknown dual-recording fly {name}")
    return ("ephys_B", "ephys_A") if name == "03" else ("ephys_A", "ephys_B")


def _recording(
    path: Path, name: str, spike_tools: Any,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    data = sio.loadmat(path, variable_names=(
        "ephys_A", "ephys_B", "ephys_SR", "ball_SR", "stim", "yaw", "fwd"
    ))
    fs, bs = int(data["ephys_SR"].item()), int(data["ball_SR"].item())
    n, step = len(data["yaw"]), fs // bs
    if fs % bs or bs != 100 or n % 5 or any(
        len(data[key].reshape(-1)) != n * step for key in ("ephys_A", "ephys_B")
    ):
        raise ValueError("unsupported ephys/ball sampling alignment")
    rates = []
    for key, prominence in zip(_channel_keys(name), (15.0, 5.75), strict=True):
        processed = spike_tools.process_voltage_spike_detection(data[key].reshape(-1), fs)
        spikes = spike_tools.detect_spikes(
            processed, method="findpeaks", window_len=10 * fs, prominence=prominence
        )
        binned = spikes.reshape(n, step).sum(axis=1).astype(float) * bs
        rates.append(gaussian_filter1d(binned, 2.5).reshape(-1, 5).mean(axis=1))
        del processed, spikes, binned
    yaw = data["yaw"].reshape(-1).reshape(-1, 5).mean(axis=1)
    forward = data["fwd"].reshape(-1).reshape(-1, 5)
    fwd, fwd_std = forward.mean(axis=1), forward.std(axis=1)
    stim = data["stim"].reshape(n, step).max(axis=1) > 0
    bad = binary_dilation(stim, iterations=25).reshape(-1, 5).any(axis=1)
    return (
        np.asarray(rates[0], dtype=np.float64),
        np.asarray(rates[1], dtype=np.float64),
        np.asarray(yaw, dtype=np.float64),
        np.asarray(fwd, dtype=np.float64),
        np.asarray(fwd_std, dtype=np.float64),
        np.asarray(bad, dtype=np.bool_),
    )


def _score(
    a2: FloatArray, a1: FloatArray, yaw: FloatArray, fwd: FloatArray,
    fwd_std: FloatArray, bad: BoolArray, *, null_draws: int = NULL_DRAWS,
    null_seed: int = NULL_SEED,
) -> dict[str, Any]:
    sizes = {len(value) for value in (a2, a1, yaw, fwd, fwd_std, bad)}
    if len(sizes) != 1 or len(yaw) < 1000:
        raise ValueError("aligned recording must contain at least 1000 bins")
    predictor_n = len(yaw) - LAG_BINS
    split = int(0.6 * predictor_n)
    indices = np.arange(predictor_n)
    valid = (fwd_std[:-LAG_BINS] > 0.01) & ~bad[:-LAG_BINS] & ~bad[LAG_BINS:]
    train = valid & (indices < split - GAP_BINS)
    test = valid & (indices >= split + GAP_BINS)
    if int(train.sum()) < 500 or int(test.sum()) < 500:
        raise ValueError("too few valid training or testing bins")
    target = yaw[LAG_BINS:]
    movement = np.column_stack((yaw[:-LAG_BINS], fwd[:-LAG_BINS]))
    a1_source, a2_source = a1[:-LAG_BINS], a2[:-LAG_BINS]

    def fit(
        neural: FloatArray | None = None, *, a1_included: bool = False
    ) -> tuple[float, list[float]]:
        columns = [movement]
        if a1_included:
            columns.append(a1_source)
        if neural is not None:
            columns.append(neural)
        x = np.column_stack(columns)
        prediction, beta = _fit_predict(x[train], target[train], x[test])
        return _mse(prediction, target[test]), [float(v) for v in beta]

    movement_mse, movement_beta = fit()
    a1_mse, a1_beta = fit(a1_included=True)
    a2_mse, a2_beta = fit(a2_source)
    full_mse, full_beta = fit(a2_source, a1_included=True)
    if min(movement_mse, a1_mse, a2_mse) <= 0:
        raise ValueError("comparator test MSE must be positive")
    effect = (a1_mse - full_mse) / a1_mse
    reciprocal = (a2_mse - full_mse) / a2_mse
    rng = np.random.default_rng(null_seed)
    shifts = rng.integers(GAP_BINS, predictor_n - GAP_BINS, size=null_draws)
    null_effects = []
    for shift in shifts:
        shifted_mse, _ = fit(np.roll(a2_source, int(shift)), a1_included=True)
        null_effects.append(float((a1_mse - shifted_mse) / a1_mse))
    p = (1 + sum(value >= effect for value in null_effects)) / (null_draws + 1)
    return {
        "total_50ms_bins": len(yaw),
        "split_index": split,
        "gap_bins_each_side": GAP_BINS,
        "train_valid_bins": int(train.sum()),
        "test_valid_bins": int(test.sum()),
        "mean_dna02_rate_hz": float(a2.mean()),
        "mean_dna01_rate_hz": float(a1.mean()),
        "movement_mse": movement_mse,
        "movement_coefficients_standardized": movement_beta,
        "dna01_mse": a1_mse,
        "dna01_coefficients_standardized": a1_beta,
        "dna02_mse": a2_mse,
        "dna02_coefficients_standardized": a2_beta,
        "full_mse": full_mse,
        "full_coefficients_standardized": full_beta,
        "dna02_incremental_relative_mse_reduction": float(effect),
        "dna01_reciprocal_relative_mse_reduction": float(reciprocal),
        "null_seed": null_seed,
        "null_shift_range_bins": [GAP_BINS, predictor_n - GAP_BINS],
        "null_shifts": [int(s) for s in shifts],
        "null_draws": null_draws,
        "null_effects": null_effects,
        "shift_null_one_sided_p": float(p),
    }


def audit(data_dir: Path, code_dir: Path, primary_code_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    revisions = []
    for source_dir in (code_dir, primary_code_dir):
        revisions.append(subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip())
    if revisions != [SECONDARY_CODE_REVISION, PRIMARY_CODE_REVISION]:
        raise ValueError("authors' analysis-code revision mismatch")
    sys.path.insert(0, str(code_dir.resolve()))
    spike_tools = import_module("a2lib.spike_tools")
    recordings: dict[str, dict[str, Any]] = {}
    for name, (file_id, expected_md5) in FILES.items():
        path = data_dir / f"a1-a2-dual-{name}.mat"
        source_digest = _digest(path)
        if source_digest["md5"] != expected_md5:
            raise ValueError(f"recording {name} Dataverse MD5 mismatch")
        a2, a1, yaw, fwd, fwd_std, bad = _recording(path, name, spike_tools)
        recordings[name] = {
            "dataverse_file_id": file_id,
            "source": source_digest,
            "channels": {"dna02": _channel_keys(name)[0], "dna01": _channel_keys(name)[1]},
            "result": _score(a2, a1, yaw, fwd, fwd_std, bad),
        }
        print(f"processed a1_a2_d_{name}", file=sys.stderr, flush=True)
    results = [recordings[name]["result"] for name in FILES]
    effects = [r["dna02_incremental_relative_mse_reduction"] for r in results]
    positive = sum(effect > 0 for effect in effects)
    significant = sum(r["shift_null_one_sided_p"] <= 0.05 for r in results)
    median = float(np.median(effects))
    gate = positive == 4 and significant >= 3 and median >= 0.01
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "software_revision": _software_revision(),
        "dataverse_doi": "10.7910/DVN/0NCLP1",
        "dataverse_version": "1",
        "authors_secondary_code_revision": revisions[0],
        "authors_primary_code_revision": revisions[1],
        "lag_ms": 150,
        "dna02_spike_prominence": 15.0,
        "dna01_spike_prominence": 5.75,
        "null_draws": NULL_DRAWS,
        "null_seed": NULL_SEED,
        "recordings": recordings,
        "summary": {
            "positive_test_effects": positive,
            "shift_null_p_at_most_0_05": significant,
            "median_relative_mse_reduction": median,
            "complementary_dna02_gate_passed": gate,
        },
        "causal_claim_allowed": False,
        "relay_physiology_claim_allowed": False,
        "behavioral_autonomy_claim_allowed": False,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--primary-code-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = audit(args.data_dir, args.code_dir, args.primary_code_dir)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
