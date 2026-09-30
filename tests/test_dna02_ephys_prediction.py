"""Synthetic checks for the locked real-fly prediction assay."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.dna02_ephys_prediction import _fit_predict, _score


def _synthetic_recording(
    seed: int = 41, n: int = 2200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=n)
    fwd = rng.normal(size=n)
    yaw = np.zeros(n, dtype=np.float64)
    yaw[:3] = rng.normal(size=3)
    for t in range(n - 3):
        yaw[t + 3] = 0.55 * yaw[t] + 0.25 * fwd[t] + 0.9 * delta[t] + rng.normal(
            scale=0.2
        )
    left = rng.normal(scale=0.1, size=n)
    right = left + delta
    fwd_std = np.full(n, 0.02)
    bad = np.zeros(n, dtype=np.bool_)
    return left, right, yaw, fwd, fwd_std, bad


def test_standardization_uses_training_only() -> None:
    train_x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    train_y = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    test_x = np.array([[1000.0], [-1000.0]], dtype=np.float64)
    prediction, beta = _fit_predict(train_x, train_y, test_x)
    np.testing.assert_allclose(prediction, [2001.0, -1999.0])
    np.testing.assert_allclose(beta, [3.0, 2.0 * np.std(train_x)])


def test_split_gap_and_incremental_score() -> None:
    recording = _synthetic_recording()
    result = _score(*recording, null_draws=19)
    split = int(0.6 * (len(recording[2]) - 3))
    assert result["split_index"] == split
    assert result["train_valid_bins"] == split - 100
    assert result["test_valid_bins"] == len(recording[2]) - 3 - split - 100
    assert result["relative_mse_reduction"] > 0.5
    assert result["enhanced_coefficients_standardized"][-1] > 0
    assert result["enhanced_mse"] < result["baseline_mse"]

    # A late-only perturbation may change test scores but must not change training fits.
    changed = list(recording)
    changed_yaw = recording[2].copy()
    changed_yaw[split + 200 :] += 10.0
    changed[2] = changed_yaw
    shifted = _score(*changed, null_draws=19)
    np.testing.assert_allclose(
        shifted["enhanced_coefficients_standardized"],
        result["enhanced_coefficients_standardized"],
    )


def test_invalid_bins_excluded_from_both_splits() -> None:
    recording = list(_synthetic_recording())
    fwd_std = recording[4].copy()
    bad = recording[5].copy()
    fwd_std[30] = 0.01
    bad[40] = True
    bad[1500] = True
    recording[4] = fwd_std
    recording[5] = bad
    result = _score(*recording, null_draws=1)
    assert result["train_valid_bins"] == result["split_index"] - 100 - 3
    assert result["test_valid_bins"] == len(recording[2]) - 3 - result["split_index"] - 100 - 2


def test_temporal_null_is_deterministic() -> None:
    recording = _synthetic_recording()
    first = _score(*recording, null_draws=19, null_seed=20260930)
    second = _score(*recording, null_draws=19, null_seed=20260930)
    assert first["null_effects"] == second["null_effects"]
    assert first["shift_null_one_sided_p"] == second["shift_null_one_sided_p"]
    assert first["null_effects"] != _score(*recording, null_draws=19, null_seed=3)[
        "null_effects"
    ]
    assert 0 < first["shift_null_one_sided_p"] <= 1


def test_rejects_zero_training_variance() -> None:
    train_x = np.ones((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="zero or invalid training variance"):
        _fit_predict(train_x, np.arange(3, dtype=np.float64), train_x)
