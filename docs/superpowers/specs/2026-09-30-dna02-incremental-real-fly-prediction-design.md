# Locked analysis: incremental DNa02 steering prediction in real flies

## Question and scope

In each of the four published dual-DNa02 recordings from living, walking
flies, does the right-minus-left DNa02 firing-rate difference at time `t`
improve prediction of yaw velocity at `t + 150 ms` beyond the information
already in yaw and forward velocity at `t`? This is a new project-level
out-of-time predictive test, not a new animal experiment or evidence that
the named premotor relays cause the effect. The same recordings and the
150-ms lag were already inspected in an earlier correlation reanalysis;
the late time periods are **not** wholly unseen biological data.

The alternative of reanalyzing Yang et al.'s direct DNa02 perturbation
data is unavailable from a public raw-data archive: the paper says the
data will be shared by the lead contact upon request. More connectome
or simulator analysis would not test living physiology. The accessible
Rayshubskiy et al. recordings are therefore the immediate source.

## Frozen source and preprocessing

Use Harvard Dataverse DOI `10.7910/DVN/0NCLP1`, version 1, file IDs
`11634638`, `11634639`, `11634640`, `11634641`, respectively `a2_d_08`,
`a2_d_12`, `a2_d_13`, `a2_d_14`. Require source MD5 values in that order:
`f3d0d40d7435af8f3d4e73d004bd3f9d`,
`71ed13ab29ecd3b3abcf7bf6b77940e4`,
`c3f95c76e9efc1a5b2277e7ee958143e`,
`eb0950ad281d61c4473404ea126b386a`.
Use the authors' secondary spike-detection code at
`7e2895349266b5cc5fa1bf53ad56e8ecc6c842e8`.

Keep the exact prior reanalysis preprocessing: `process_voltage_spike_detection`
and `detect_spikes(method="findpeaks", prominence=15.0,
window_len=10*ephys_SR)` per channel; bin spikes at 100 Hz; smooth with
a Gaussian of sigma 2.5 samples; average into 50-ms bins. Define neural
`d(t) = right(t) - left(t)`, yaw `y(t)`, and forward velocity `f(t)` as
50-ms means. Keep bins with within-bin 100-Hz forward-velocity standard
deviation above `0.01`. Exclude stimulus-containing 100-Hz bins and
250 ms on both sides; both predictor and outcome bins must be outside
that exclusion. Target `Y(t) = y(t+3)`; do not fit or tune the lag.
The Gaussian smoother is symmetric, so this is offline temporal prediction,
not a strictly causal real-time decoder.

Split each fly by **wall-clock index** at 60% of the eligible `t` range,
before filtering valid bins. Use `t < split - 100` for training and
`t >= split + 100` for testing, leaving at least five seconds on both
sides of the boundary. Do not shuffle or reassign bins. Require at least
500 valid bins in each split. Fit ordinary least squares on training bins
with an intercept. Standardize each predictor using **training-only**
mean and standard deviation; reject a zero-variance predictor.

Baseline predictors: `[y(t), f(t)]`. Enhanced predictors:
`[y(t), f(t), d(t)]`. An auxiliary directional comparator replaces
`d(t)` with the bilateral sum `right(t) + left(t)`; it is descriptive,
not a gate. Score mean squared error on the frozen testing bins.
Primary effect = `(MSE_baseline - MSE_enhanced) / MSE_baseline`.
Also report the trained coefficient sign of `d(t)` and the number of
training/testing bins per fly.

For a within-fly temporal null, circularly shift the complete neural
`d(t)` series 999 times with a fixed RNG seed `20260930`, choosing shifts
uniformly from integer offsets `[100, len(d)-100)`. For each shift,
refit the enhanced model on the same training bins and score on the
same testing bins. Keep baseline unchanged. The one-sided Monte Carlo
p-value is `(1 + count(null_effect >= observed_effect)) / 1000`.
This null controls alignment with yaw, not every shared cause of spiking
and behavior; time bins are not independent animals.

## Predeclared interpretation gate

Call this a positive **within-recording incremental prediction** result
only if all four test-set effects are positive, at least three of four
shift-null p-values are `<= 0.05`, the median relative MSE reduction is
at least `0.01`, and all four trained neural coefficients have the
previously observed positive sign. Otherwise report the per-fly outcomes
as negative or mixed without changing preprocessing, split, predictors,
or threshold. Four animals are too few to claim population-level
significance from a 4/4 sign pattern (`p=0.0625` one-sided).

Retain source and code hashes, exact split indices, valid counts, model
coefficients, MSE values, null seed/range/p-values, software revision,
runtime and a machine-readable artifact. Unit-test split isolation,
training-only standardization, baseline/enhanced scoring, and null
determinism on synthetic data before running the real recordings.

Even a passing result would show information in DNa02 about future
steering beyond the chosen movement-history baseline. It would not
establish DNa02 causation, relay physiology, muscle force, or autonomous
learning in FlyBrain. Existing direct optogenetic work supports a
broader causal role of DNa02 in leg movements, but is external literature
and does not validate this particular model circuit.
