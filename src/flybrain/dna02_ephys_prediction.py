"""Out-of-time incremental DNa02 steering prediction in published fly recordings."""

from __future__ import annotations

import argparse
import hashlib
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

from flybrain.mb_association import _software_revision

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
PROTOCOL = "dna02-real-ephys-incremental-prediction-v1"
DESIGN_REVISION = "930ad18"
CODE_REVISION = "7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8"
FILES = {
    "08": "f3d0d40d7435af8f3d4e73d004bd3f9d",
    "12": "71ed13ab29ecd3b3abcf7bf6b77940e4",
    "13": "c3f95c76e9efc1a5b2277e7ee958143e",
    "14": "eb0950ad281d61c4473404ea126b386a",
}
LAG_BINS = 3
GAP_BINS = 100
NULL_DRAWS = 999
NULL_SEED = 20260930


def _digest(path: Path) -> dict[str, Any]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def _fit_predict(
    train_x: FloatArray, train_y: FloatArray, test_x: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    if not bool(np.all(np.isfinite(scale))) or bool(np.any(scale <= 0)):
        raise ValueError("predictor has zero or invalid training variance")
    x_train = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    x_test = np.column_stack((np.ones(len(test_x)), (test_x - mean) / scale))
    beta, _, rank, _ = np.linalg.lstsq(x_train, train_y, rcond=None)
    if rank != x_train.shape[1]:
        raise ValueError("predictors are linearly dependent in training data")
    return np.asarray(x_test @ beta, dtype=np.float64), np.asarray(beta, dtype=np.float64)


def _mse(pred: FloatArray, actual: FloatArray) -> float:
    return float(np.mean(np.square(pred - actual)))


def _score(
    left: FloatArray, right: FloatArray, yaw: FloatArray, fwd: FloatArray,
    fwd_std: FloatArray, bad: BoolArray, *, null_draws: int = NULL_DRAWS,
    null_seed: int = NULL_SEED,
) -> dict[str, Any]:
    sizes = {len(value) for value in (left, right, yaw, fwd, fwd_std, bad)}
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
    baseline_x = np.column_stack((yaw[:-LAG_BINS], fwd[:-LAG_BINS]))
    baseline_pred, baseline_beta = _fit_predict(
        baseline_x[train], target[train], baseline_x[test]
    )
    baseline_mse = _mse(baseline_pred, target[test])
    if baseline_mse <= 0:
        raise ValueError("baseline test MSE must be positive")
    delta = (right - left)[:-LAG_BINS]
    bilateral_sum = (right + left)[:-LAG_BINS]

    def enhanced(neural: FloatArray) -> tuple[float, list[float]]:
        x = np.column_stack((baseline_x, neural))
        pred, beta = _fit_predict(x[train], target[train], x[test])
        return _mse(pred, target[test]), [float(value) for value in beta]

    enhanced_mse, enhanced_beta = enhanced(delta)
    sum_mse, sum_beta = enhanced(bilateral_sum)
    effect = (baseline_mse - enhanced_mse) / baseline_mse
    if predictor_n <= 2 * GAP_BINS:
        raise ValueError("recording is too short for null shifts")
    rng = np.random.default_rng(null_seed)
    shifts = rng.integers(GAP_BINS, predictor_n - GAP_BINS, size=null_draws)
    null = []
    for shift in shifts:
        shifted_mse, _ = enhanced(np.roll(delta, int(shift)))
        null.append(float((baseline_mse - shifted_mse) / baseline_mse))
    p = (1 + sum(value >= effect for value in null)) / (null_draws + 1)
    return {
        "total_50ms_bins": len(yaw),
        "split_index": split,
        "gap_bins_each_side": GAP_BINS,
        "train_valid_bins": int(train.sum()),
        "test_valid_bins": int(test.sum()),
        "baseline_mse": baseline_mse,
        "baseline_coefficients_standardized": [float(v) for v in baseline_beta],
        "enhanced_mse": enhanced_mse,
        "enhanced_coefficients_standardized": enhanced_beta,
        "relative_mse_reduction": float(effect),
        "bilateral_sum_mse": sum_mse,
        "bilateral_sum_coefficients_standardized": sum_beta,
        "bilateral_sum_relative_mse_reduction": float((baseline_mse - sum_mse) / baseline_mse),
        "null_seed": null_seed,
        "null_shift_range_bins": [GAP_BINS, predictor_n - GAP_BINS],
        "null_draws": null_draws,
        "null_effects": null,
        "shift_null_one_sided_p": float(p),
    }


def _recording(
    path: Path, spike_tools: Any,
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
    for key in ("ephys_A", "ephys_B"):
        processed = spike_tools.process_voltage_spike_detection(data[key].reshape(-1), fs)
        spikes = spike_tools.detect_spikes(
            processed, method="findpeaks", window_len=10 * fs, prominence=15.0
        )
        binned = spikes.reshape(n, step).sum(axis=1).astype(float) * bs
        rates.append(gaussian_filter1d(binned, 2.5).reshape(-1, 5).mean(axis=1))
        del processed, spikes, binned
    yaw = data["yaw"].reshape(-1).reshape(-1, 5).mean(axis=1)
    forward = data["fwd"].reshape(-1).reshape(-1, 5)
    fwd, fwd_std = forward.mean(axis=1), forward.std(axis=1)
    stim = data["stim"].reshape(n, step).max(axis=1) > 0
    bad = binary_dilation(stim, iterations=25).reshape(-1, 5).any(axis=1)
    return (np.asarray(rates[0], dtype=np.float64), np.asarray(rates[1], dtype=np.float64),
            np.asarray(yaw, dtype=np.float64), np.asarray(fwd, dtype=np.float64),
            np.asarray(fwd_std, dtype=np.float64), np.asarray(bad, dtype=np.bool_))


def audit(data_dir: Path, code_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    code_revision = subprocess.check_output(
        ["git", "-C", str(code_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_revision != CODE_REVISION:
        raise ValueError("authors' spike-processing revision mismatch")
    sys.path.insert(0, str(code_dir.resolve()))
    spike_tools = import_module("a2lib.spike_tools")
    recordings = {}
    for name, expected_md5 in FILES.items():
        path = data_dir / f"a2-dual-{name}.mat"
        source = _digest(path)
        if source["md5"] != expected_md5:
            raise ValueError(f"recording {name} Dataverse MD5 mismatch")
        left, right, yaw, fwd, fwd_std, bad = _recording(path, spike_tools)
        recordings[name] = {"source": source,
                            "result": _score(left, right, yaw, fwd, fwd_std, bad)}
        print(f"processed a2_d_{name}", file=sys.stderr, flush=True)
    results = [recordings[name]["result"] for name in FILES]
    positive_count = sum(r["relative_mse_reduction"] > 0 for r in results)
    significant_count = sum(r["shift_null_one_sided_p"] <= 0.05 for r in results)
    median_effect = float(np.median([r["relative_mse_reduction"] for r in results]))
    positive_coefficients = all(r["enhanced_coefficients_standardized"][-1] > 0
                                for r in results)
    gate = (positive_count == 4 and significant_count >= 3
            and median_effect >= 0.01 and positive_coefficients)
    return {
        "protocol": PROTOCOL,
        "design_revision": DESIGN_REVISION,
        "software_revision": _software_revision(),
        "dataverse_doi": "10.7910/DVN/0NCLP1",
        "dataverse_version": "1",
        "authors_code_revision": code_revision,
        "null_draws": NULL_DRAWS,
        "null_seed": NULL_SEED,
        "lag_ms": 150,
        "recordings": recordings,
        "summary": {
            "positive_test_effects": positive_count,
            "shift_null_p_at_most_0_05": significant_count,
            "median_relative_mse_reduction": median_effect,
            "positive_neural_coefficients": positive_coefficients,
            "incremental_prediction_gate_passed": gate,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = audit(args.data_dir, args.code_dir)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
