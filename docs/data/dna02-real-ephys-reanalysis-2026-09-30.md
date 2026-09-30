# DNa02 steering relation in real-fly electrophysiology

This is a project-level reanalysis of **published recordings from living,
walking flies**, not a new animal experiment or a novel claim about the
DNa02 circuit. It tests whether the biological steering signal invoked by
FlyBrain is visible in public raw data. It does not test the simulator's
motor decoder or the specific DNa02→IN19A003/IN08A006→leg-rotator synapses.
The analysis was exploratory: its precise combined pipeline was not committed
before the data were inspected. Session `a2_d_13` was used to diagnose an
initial zero-lag result; all final parameters were then applied unchanged to
the other three dual-DNa02 sessions.

## Primary sources and provenance

- [Rayshubskiy et al., *eLife* 2025](https://doi.org/10.7554/eLife.102230)
  reported that the right-minus-left DNa02 firing-rate difference predicts
  rotational velocity during walking. This reanalysis is not independent of
  that published discovery.
- The [Harvard Dataverse dataset](https://doi.org/10.7910/DVN/0NCLP1),
  version 1, contains the four dual-DNa02 `.mat` recordings below. Only
  these four files were downloaded for this audit; the complete deposit is
  about 158 GB and includes many unrelated images.
- Spike processing uses the authors' [secondary Python code](https://github.com/wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code)
  at commit `7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`. The 50-ms
  analysis window, movement filter, and 150-ms response lag follow the
  [authors' MATLAB analysis](https://github.com/SashaRayshubskiy/eLife_102230_analysis_code)
  at commit `55e30c19b1a18f601df1803295f0c401aec3c167`. This hybrid is
  not a byte-for-byte rerun of their paper figure.

| Session | Dataverse file ID | MD5 | Recorded seconds |
| --- | ---: | --- | ---: |
| `a2_d_08` | 11634638 | `f3d0d40d7435af8f3d4e73d004bd3f9d` | 3,450 |
| `a2_d_12` | 11634639 | `71ed13ab29ecd3b3abcf7bf6b77940e4` | 1,495 |
| `a2_d_13` | 11634640 | `c3f95c76e9efc1a5b2277e7ee958143e` | 575 |
| `a2_d_14` | 11634641 | `eb0950ad281d61c4473404ea126b386a` | 1,449 |

The authors' metadata identifies `ephys_A` as left DNa02 and `ephys_B` as
right DNa02. All four file MD5 values matched Dataverse. The additional
single-cell recording `a2_s_03` was acquired for format inspection but was
not included in this bilateral analysis.
Local SHA-256 values for `08`, `12`, `13`, and `14`, in that order, are
`f02a345effb2fe722800dd1f43dce2c876878424307cb1fa486b5974f3d5922b`,
`0f9a9251262f528f0b415d67cdb169bc1236a5157029ffeebace6d86136f60f7`,
`a460eb313b503b039b3eeec554efd902b54119aadfd783c72210efaee2daee24`,
and `6e5aabc4bc45d76fd24baa81a9351cc6d29530c72e697038b78f07a0da36623f`.

## Fixed calculation and result

For each channel, use the published Python `process_voltage_spike_detection`
and `detect_spikes(method="findpeaks", prominence=15.0,
window_len=10*ephys_SR)`. Bin at the ball's 100-Hz sample rate; estimate
firing rate by Gaussian smoothing with sigma 25 ms; average into 50-ms bins.
Keep bins where the forward-velocity standard deviation exceeds `0.01`, as
in the authors' MATLAB example. Exclude stimulus bins and 250 ms on either
side. Correlate right-minus-left firing rate at bin `t` with yaw at `t+150 ms`.
No parameter was refit per fly. The zero-lag correlation is a temporal
comparator. For a within-recording null, circularly shift the neural series
999 times with shifts at least five seconds from zero (fixed RNG seed
`20260930`), retaining the same valid-bin mask. Its two-sided Monte Carlo
minimum p-value is `0.001`; correlated time bins are **not** independent
animal replicates.

| Fly | Valid 50-ms bins | r at 0 ms | r at +150 ms | Shift-null p |
| --- | ---: | ---: | ---: | ---: |
| `a2_d_08` | 37,343 | 0.1773 | 0.3821 | 0.001 |
| `a2_d_12` | 18,028 | 0.4150 | 0.6440 | 0.001 |
| `a2_d_13` | 5,735 | 0.0660 | 0.3596 | 0.001 |
| `a2_d_14` | 13,464 | 0.2294 | 0.5088 | 0.001 |

All four recordings show the reported sign, with median `r=0.4455` at
+150 ms. Every value exceeds the 95th percentile of the absolute
circular-shift null in its own recording. The animal is the replication unit:
with only four flies, a one-sided sign test for 4/4 positive has `p=0.0625`.
The shift-null values alone must not be reported as an across-animal
significance test.

## Reproduce and limit the claim

From the FlyBrain root, obtain only the four Dataverse files and the
authors' MIT-licensed analysis code. Confirm the four MD5 values against the
table above before analysis:

```bash
mkdir -p artifacts/rayshubskiy-ephys-external
curl -L --fail 'https://dataverse.harvard.edu/api/access/datafile/11634638' -o artifacts/rayshubskiy-ephys-external/a2-dual-08.mat
curl -L --fail 'https://dataverse.harvard.edu/api/access/datafile/11634639' -o artifacts/rayshubskiy-ephys-external/a2-dual-12.mat
curl -L --fail 'https://dataverse.harvard.edu/api/access/datafile/11634640' -o artifacts/rayshubskiy-ephys-external/a2-dual-13.mat
curl -L --fail 'https://dataverse.harvard.edu/api/access/datafile/11634641' -o artifacts/rayshubskiy-ephys-external/a2-dual-14.mat
md5 artifacts/rayshubskiy-ephys-external/a2-dual-*.mat
git clone https://github.com/wilson-lab/rayshubskiy_elife_102230_secondary_analysis_code.git artifacts/rayshubskiy-code-external
git -C artifacts/rayshubskiy-code-external checkout 7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8
```

Run the calculation below (the extra `tqdm` is used transiently and does
not alter FlyBrain's lockfile):

```bash
PYTHONPATH=artifacts/rayshubskiy-code-external uv run --with tqdm python - <<'PY'
import json
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation, gaussian_filter1d
from a2lib.spike_tools import detect_spikes, process_voltage_spike_detection

for name in ('08', '12', '13', '14'):
    path = f'artifacts/rayshubskiy-ephys-external/a2-dual-{name}.mat'
    d = sio.loadmat(path, variable_names=(
        'ephys_A', 'ephys_B', 'ephys_SR', 'ball_SR', 'stim', 'yaw', 'fwd'))
    fs, bs = int(d['ephys_SR'].item()), int(d['ball_SR'].item())
    n, step = len(d['yaw']), fs // bs
    assert fs % bs == 0 and len(d['ephys_A']) == len(d['ephys_B']) == n * step
    rates = []
    for key in ('ephys_A', 'ephys_B'):
        processed = process_voltage_spike_detection(d[key].reshape(-1), fs)
        spikes = detect_spikes(processed, method='findpeaks',
                               window_len=10 * fs, prominence=15.0)
        binned = spikes.reshape(n, step).sum(axis=1).astype(float) * bs
        rates.append(gaussian_filter1d(binned, 2.5))
    aggregate = lambda x: x.reshape(-1, 5).mean(axis=1)
    left, right = map(aggregate, rates)
    yaw = aggregate(d['yaw'].reshape(-1))
    fwd_std = d['fwd'].reshape(-1).reshape(-1, 5).std(axis=1)
    stim = d['stim'].reshape(n, step).max(axis=1) > 0
    bad = binary_dilation(stim, iterations=25).reshape(-1, 5).any(axis=1)
    lag = 3
    valid = (fwd_std[:-lag] > 0.01) & ~bad[:-lag] & ~bad[lag:]
    neural, future_yaw = (right - left)[:-lag], yaw[lag:]
    observed = float(np.corrcoef(neural[valid], future_yaw[valid])[0, 1])
    zero_lag = float(np.corrcoef(neural[valid], yaw[:-lag][valid])[0, 1])
    rng = np.random.default_rng(20260930)
    shifts = rng.integers(100, len(neural) - 100, size=999)
    null = np.array([np.corrcoef(np.roll(neural, int(k))[valid],
                                 future_yaw[valid])[0, 1] for k in shifts])
    p = (1 + int((np.abs(null) >= abs(observed)).sum())) / 1000
    print(json.dumps({'fly': name, 'valid_bins': int(valid.sum()),
                      'r_0ms': zero_lag, 'r_150ms': observed, 'shift_p': p}))
PY
```

This verifies an observational DNa02–steering association in real flies and
provides an empirical target for the biological model. It does not establish
that DNa02 spikes cause a particular leg-rotator response, that the named
premotor synapses have an inhibitory postsynaptic effect, or that FlyBrain
reproduces these physiological trajectories. Those claims need matched
interventions or direct simultaneous measurements in the living circuit.
The [BANC muscle-route anatomy](banc-dna02-rotator-anatomy-2026-09-30.md)
was measured in a different animal; combining the two datasets does not
turn a structural route into a measured functional pathway.
