"""Synthetic checks for the locked same-fly dual-descending-neuron assay."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.dna01_dna02_paired_prediction import _channel_keys, _recording, _score


def _synthetic(
    seed: int = 99, n: int = 2200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a1 = rng.normal(size=n)
    a2 = 0.4 * a1 + rng.normal(size=n)
    fwd = rng.normal(size=n)
    yaw = np.zeros(n, dtype=np.float64)
    yaw[:3] = rng.normal(size=3)
    for t in range(n - 3):
        yaw[t + 3] = (
            0.5 * yaw[t] + 0.2 * fwd[t] + 0.3 * a1[t] + a2[t]
            + rng.normal(scale=0.25)
        )
    return (
        a2, a1, yaw, fwd, np.full(n, 0.02), np.zeros(n, dtype=np.bool_)
    )


def test_documented_channel_swap() -> None:
    assert _channel_keys("01") == ("ephys_A", "ephys_B")
    assert _channel_keys("02") == ("ephys_A", "ephys_B")
    assert _channel_keys("03") == ("ephys_B", "ephys_A")
    assert _channel_keys("04") == ("ephys_A", "ephys_B")
    with pytest.raises(ValueError, match="unknown dual-recording"):
        _channel_keys("05")


def test_recording_applies_the_channel_swap_and_fixed_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from flybrain import dna01_dna02_paired_prediction as paired

    n, fs = 10, 10000
    raw = {
        "ephys_A": np.full((n * 100, 1), 10.0),
        "ephys_B": np.full((n * 100, 1), 20.0),
        "ephys_SR": np.array([[fs]]),
        "ball_SR": np.array([[100]]),
        "stim": np.zeros((1, n * 100)),
        "yaw": np.zeros((n, 1)),
        "fwd": np.zeros((n, 1)),
    }
    monkeypatch.setattr(paired.sio, "loadmat", lambda *_args, **_kwargs: raw)
    calls: list[tuple[float, float]] = []

    def detect_spikes(
        signal: np.ndarray, *, method: str, window_len: int, prominence: float
    ) -> np.ndarray:
        assert method == "findpeaks"
        assert window_len == 10 * fs
        calls.append((float(signal[0]), prominence))
        return np.zeros(len(signal))

    spike_tools = SimpleNamespace(
        process_voltage_spike_detection=lambda signal, _fs: signal,
        detect_spikes=detect_spikes,
    )
    _recording(Path("unused.mat"), "01", spike_tools)
    assert calls == [(10.0, 15.0), (20.0, 5.75)]
    calls.clear()
    _recording(Path("unused.mat"), "03", spike_tools)
    assert calls == [(20.0, 15.0), (10.0, 5.75)]


def test_complementary_score_and_split_isolation() -> None:
    recording = _synthetic()
    result = _score(*recording, null_draws=19)
    split = int(0.6 * (len(recording[2]) - 3))
    assert result["split_index"] == split
    assert result["train_valid_bins"] == split - 100
    assert result["test_valid_bins"] == len(recording[2]) - 3 - split - 100
    assert result["dna02_incremental_relative_mse_reduction"] > 0.5
    assert result["dna01_reciprocal_relative_mse_reduction"] > 0.05
    assert result["full_mse"] < result["dna01_mse"]
    assert result["full_mse"] < result["dna02_mse"]

    changed = list(recording)
    changed_yaw = recording[2].copy()
    changed_yaw[split + 200 :] += 10.0
    changed[2] = changed_yaw
    late_perturbed = _score(*changed, null_draws=19)
    np.testing.assert_allclose(
        late_perturbed["full_coefficients_standardized"],
        result["full_coefficients_standardized"],
    )


def test_mask_and_null_determinism() -> None:
    recording = list(_synthetic())
    fwd_std = recording[4].copy()
    bad = recording[5].copy()
    fwd_std[30] = 0.01
    bad[40] = True
    bad[1500] = True
    recording[4] = fwd_std
    recording[5] = bad
    first = _score(*recording, null_draws=19, null_seed=20260930)
    second = _score(*recording, null_draws=19, null_seed=20260930)
    assert first["train_valid_bins"] == first["split_index"] - 100 - 3
    assert first["test_valid_bins"] == len(recording[2]) - 3 - first["split_index"] - 100 - 2
    assert first["null_shifts"] == second["null_shifts"]
    assert first["null_effects"] == second["null_effects"]
    assert first["shift_null_one_sided_p"] == second["shift_null_one_sided_p"]
    assert first["null_shifts"] != _score(*recording, null_draws=19, null_seed=3)[
        "null_shifts"
    ]
