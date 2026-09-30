# Locked analysis: DNa02 information beyond simultaneous DNa01 in real flies

## Scope and choice

Test whether DNa02 contributes out-of-time information about steering when
DNa01 is recorded at the same moment in the same walking fly. Both neurons are
left-hemisphere descending cells. This is a new FlyBrain reanalysis of public
physiology, not a new animal experiment or a test of the named VNC relays.
The published study already showed different DNa01/DNa02 steering dynamics;
these same recordings have been inspected by their authors, so this is not
independent discovery. The previously chosen 150-ms lag and processing scheme
were also inspected in FlyBrain's bilateral DNa02 analysis.

Direct Yang et al. perturbation traces are available from the lead contact on
request rather than in a public raw-data archive. The Rayshubskiy central-
complex stimulation experiments have four experimental and four control flies,
but reproducible EPG bump extraction requires large TIFF stacks and MATLAB
helpers absent from the public analysis repo. The simultaneous DNa01/DNa02
recordings are the strongest immediately reproducible physiological comparator.

## Frozen source and preprocessing

Use Harvard Dataverse DOI `10.7910/DVN/0NCLP1`, version 1, four dual recordings:

| Fly | File ID | Required MD5 |
| --- | ---: | --- |
| `a1_a2_d_01` | 11634626 | `9cfaf03120f6ae7f09f4766b8dfef3c9` |
| `a1_a2_d_02` | 11634627 | `eea3f8e609abf138c4f223d56162a196` |
| `a1_a2_d_03` | 11634628 | `0b570dce1281528e1ad032b5ec7dfda3` |
| `a1_a2_d_04` | 11634629 | `1e6301518a7c15954c26ec2743df58ec` |

Use authors' secondary spike-processing code at
`7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`. Its
`import_preprocess_data.ipynb` labels every pair `['a2_l','a1_l']`, notes that
the A1 channel in fly 02 is difficult to detect, and uses `findpeaks`
prominence 15.0 for DNa02 and 5.75 for DNa01. The authors' primary MATLAB
script `A1_A2_dual_patch_analysis_for_paper.m` at
`55e30c19b1a18f601df1803295f0c401aec3c167` notes that fly 03 has
the channels swapped. Therefore files 01, 02, 04 have DNa02=`ephys_A`,
DNa01=`ephys_B`; file 03 has DNa02=`ephys_B`, DNa01=`ephys_A`.

Per channel use `process_voltage_spike_detection`, then
`detect_spikes(method="findpeaks", window_len=10*ephys_SR, prominence=<above>)`.
Bin spikes at the ball's 100 Hz, Gaussian smooth at sigma 2.5 ball samples,
then average into 50-ms bins. Use 50-ms means of yaw and forward velocity.
Keep source bins with within-bin 100-Hz forward-velocity standard deviation
strictly above `0.01`. Exclude stimulus-containing ball bins and 250 ms on
both sides; both source and target 50-ms bins must be outside the exclusion.
Target yaw at `t+3` (150 ms); no lag search or per-fly tuning. This symmetric
Gaussian preprocessing is offline, not a causal real-time decoder.

For each fly, split by wall-clock index at 60% of the full eligible source
index range before movement/stimulus masking. Train on `t < split-100`; test
on `t >= split+100`, a 5-s gap on each side. Require at least 500 valid bins
in each. Fit OLS with intercept; standardize predictors using training-only
means and standard deviations; reject zero variance or rank deficiency.

## Frozen comparisons and gate

Baseline `B`: `[yaw(t), forward(t)]`. Primary comparison `A`: `B + DNa01(t)`
against `A + DNa02(t)`. The DNa02 incremental effect is
`(MSE_A - MSE_A+DNa02)/MSE_A` on the unchanged test bins. The reciprocal
descriptive comparison adds DNa01 to `B + DNa02`; report its relative MSE
reduction but do not use it as a causal estimate. Also report `B`, `B+DNa01`,
`B+DNa02`, and `B+DNa01+DNa02` MSEs and all standardized coefficients.

For a temporal null, circularly shift only the complete DNa02 series 999
times using RNG seed `20260930` with offsets sampled uniformly from integer
`[100, len(source_bins)-100)`. Keep DNa01, movement, target and masks fixed;
refit the full model for each shift. One-sided within-fly p-value is
`(1 + count(null_effect >= observed_effect)) / 1000`. Time bins are not
independent animal replicates.

Call a positive *within-recording complementary DNa02 signal* only if all
four primary test effects are positive, at least three of four temporal-null
p-values are at most `0.05`, and the median primary relative MSE reduction is
at least `0.01`. The reciprocal DNa01 effects are descriptive, not a gate.
The known fly-02 spike-detection difficulty must be disclosed regardless of
outcome; do not exclude it post hoc. Failure of any condition yields a mixed
or negative result under this protocol, not a retuned analysis.

Retain source MD5 and SHA-256, code revisions, all split/mask counts,
standardized coefficients, MSEs, null shifts/effects/seed/p-values, software
revision, runtime, and machine-readable output. Unit-test channel assignment,
train-only scaling, split isolation, scoring, and null determinism before
inspecting the result. Even a passing result remains observational: it does
not prove DNa02 causes steering, that a specific relay or muscle is involved,
or that FlyBrain reproduces these neural dynamics.
