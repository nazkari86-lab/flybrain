# Paired DNa01/DNa02 assay: spike-quality gate failed

The [locked protocol](../superpowers/specs/2026-09-30-dna01-dna02-paired-real-fly-design.md)
and [auditor](../../src/flybrain/dna01_dna02_paired_prediction.py) were
committed as `0e62d10` and `4df334b`, respectively, before checking the
paired-recording outcomes. This audit does **not** establish the proposed
four-fly complementary-DNa02 result. The full command stopped during the
temporal null of fly 02 because some circular shifts made a nearly silent
DNa02 predictor constant in its training interval. The predeclared OLS rule
rejects zero training variance. No primary result JSON was written.

## Sources and observed QC

All four public files are from [Harvard Dataverse DOI 10.7910/DVN/0NCLP1](https://doi.org/10.7910/DVN/0NCLP1),
version 1. Their MD5 values matched the deposit. Files remain local and are
not committed to Git.

| Fly | File ID | MD5 | SHA-256 | DNa02 mean rate at locked threshold |
| --- | ---: | --- | --- | ---: |
| 01 | 11634626 | `9cfaf03120f6ae7f09f4766b8dfef3c9` | `f53347125aba3832bb395b9d17a5bedd6cd3379d442fb45d7b076e120250c8be` | 5.099 Hz |
| 02 | 11634627 | `eea3f8e609abf138c4f223d56162a196` | `1774c88b9627ff303318c5b2708412632c41aa3b661cfa53b8aa4b883afa6c41` | 0.00966 Hz |
| 03 | 11634628 | `0b570dce1281528e1ad032b5ec7dfda3` | `ee448afcbf8239d3097262010fb15a793da139399137bfe5b6a070cfc4d26be3` | 0.02375 Hz |
| 04 | 11634629 | `1e6301518a7c15954c26ec2743df58ec` | `4fe59433276f6c68f18346c8f920a1feb8741a386bdc8459927f63c53222fadc` | 5.525 Hz |

Using only the selected authors' spike-processing code and the frozen 15.0
DNa02 / 5.75 DNa01 prominence settings, fly 02's DNa02 rate was nonzero in
only 61 of 28,980 50-ms bins; fly 03's was nonzero in only 282 of 49,680.
The normal, unshifted fly-02 fit had nonzero DNa02 training variance. The
failure arose inside the null loop when a shifted sparse trace had zero
training variance; an uncaught null-specific rank failure similarly occurred
for fly 03 in a diagnostic run. A lower DNa02 threshold of 5.75 on fly 02
detected 54,143 raw peaks versus 14 at 15.0. This is **diagnostic**, not an
authorized post hoc replacement of the locked detector.

An exploratory call with `null_draws=0` produced DNa02 incremental test-MSE
reductions of 11.29%, 0.21%, 0.26%, and 7.36% for flies 01–04, respectively.
It bypasses the required 999-shift control and therefore cannot pass the
protocol or support a four-fly significance claim. Fly 01's 99-shift
diagnostic gave `p=0.01`; this is not the frozen 999-shift result. The
authors' secondary notebook warns that DNa01 spike detection in fly 02 is
difficult. It labels file 03 as `[a2_l,a1_l]`, while the authors' primary
MATLAB script says its channels are swapped; the auditor followed the latter.
This cross-code discrepancy is an additional provenance uncertainty for 03.

## Reproduce and interpretation

After acquiring the four files and the two author-code revisions in the
protocol, run:

```bash
PYTHONPATH=artifacts/rayshubskiy-code-external uv run --with tqdm \
  python -m flybrain.dna01_dna02_paired_prediction \
  --data-dir artifacts/rayshubskiy-ephys-external \
  --code-dir artifacts/rayshubskiy-code-external \
  --primary-code-dir artifacts/rayshubskiy-primary-code-external \
  --output artifacts/dna01-dna02-paired-prediction-v1.json
```

The observed error is `predictor has zero or invalid training variance`
after `processed a1_a2_d_01`; a direct fly-02 traceback locates it in the
shift-null fit. Prior to the raw run, Ruff and mypy were clean and the full
test suite returned 518 passed / 28 skipped. These software checks do not
validate the neural spike calls. The correct status is an **invalid primary
assay under the locked detector**, not a biological negative or positive
finding about DNa02. A new voltage-based or quality-calibrated spike assay
would need an explicitly new, exploratory protocol and independent validation;
the present thresholds must not be silently changed to rescue the gate.
