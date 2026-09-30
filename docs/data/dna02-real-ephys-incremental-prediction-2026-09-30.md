# DNa02 adds out-of-time steering information in four real flies

The locked [protocol](../superpowers/specs/2026-09-30-dna02-incremental-real-fly-prediction-design.md)
passed its predeclared **within-recording incremental-prediction** gate. In each
of four published dual-DNa02 recordings, adding right-minus-left DNa02 firing
rate at time `t` to a linear model of current yaw and forward velocity reduced
test-set mean squared error for yaw at `t + 150 ms`. The out-of-time test is a
contiguous late part of the same recording, not a new animal or an unseen
biological dataset. The 150-ms lag and underlying recordings had been inspected
in an earlier exploratory analysis, so this is not fully prospective evidence.

| Fly | Training bins | Test bins | Baseline MSE | DNa02 MSE | Test MSE reduction | Shift-null p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a2_d_08` | 24,317 | 12,897 | 30,406.36 | 29,150.32 | 4.13% | 0.001 |
| `a2_d_12` | 10,660 | 7,228 | 74,423.53 | 57,159.52 | 23.20% | 0.001 |
| `a2_d_13` | 3,071 | 2,562 | 39,489.33 | 35,511.55 | 10.07% | 0.001 |
| `a2_d_14` | 7,569 | 5,779 | 47,669.46 | 32,987.83 | 30.80% | 0.001 |

The median relative MSE reduction is **16.64%**. All four fitted neural
coefficients are positive. All four one-sided within-fly circular-shift tests
reach the Monte Carlo floor, `p=0.001` with 999 shifts. This passes the frozen
gate: 4/4 positive test effects; at least 3/4 shift-null p-values at most
0.05; median reduction at least 1%; and 4/4 positive coefficients. The
bilateral-sum comparator reduced MSE by −0.18%, 6.99%, −4.48%, and −0.04%
respectively; it was descriptive, not part of the gate.

The [machine-readable result](../../artifacts/dna02-ephys-incremental-v1.json)
contains source MD5/SHA-256 hashes, author-code revision, exact split indices,
all fitted standardized coefficients, 999 null effects per fly, seed, software
revision, runtime, and peak RSS. Its SHA-256 is
`4c2a5066231f71e3d9657e37ca46f8f21b66dd2b9443ac18017e0f864a10d405`.
The four file MD5 values match Harvard Dataverse version 1, DOI
`10.7910/DVN/0NCLP1`; the authors' spike-code commit is
`7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`. The FlyBrain auditor ran
at commit `c90756bd1e9af37aa9e311bb796f154d749460fa`, in 34.242 seconds
with reported peak RSS 4,183,261,184 bytes.

Reproduce from the repository root after obtaining the four recordings and
the authors' code as described in the [earlier reanalysis](dna02-real-ephys-reanalysis-2026-09-30.md):

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
PYTHONPATH=artifacts/rayshubskiy-code-external uv run --with tqdm \
  python -m flybrain.dna02_ephys_prediction \
  --data-dir artifacts/rayshubskiy-ephys-external \
  --code-dir artifacts/rayshubskiy-code-external \
  --output artifacts/dna02-ephys-incremental-reproduction.json
```

The check before the data run returned `514 passed, 28 skipped`, with clean
Ruff and mypy results. A separate `jq` check of the output recomputed every
relative MSE reduction and null p-value from the saved arrays and confirmed
the four positive coefficient signs and gate. This arithmetic cross-check is
not an independent reprocessing of the raw voltage traces.

## What remains unproved

These are observational recordings. The shift null tests temporal alignment
inside each fly, not all behavioral confounds or across-animal significance.
Four positive animals alone yield a one-sided sign-test `p=0.0625`; time bins
must not be counted as independent animals. Gaussian smoothing is symmetric,
so the signal is suitable for offline prediction, not yet a strictly causal
real-time decoder. The result agrees with the published DNa02 steering
association and provides an empirical target for FlyBrain; it does not show
that FlyBrain itself reproduces these trajectories.

In particular, this analysis does **not** prove that DNa02 causes the measured
turns, that the BANC-identified premotor relays have the proposed functional
sign or force effect in a living fly, that the MaleCNS simulation produces
natural steering, or that its game-learning subsystem is a biologically
autonomous fly brain. Those claims require matched perturbations/direct
physiology and held-out animals or conditions, not stronger wording around
the present correlation and connectome evidence.
