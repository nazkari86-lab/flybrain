# MaleCNS Shiu plastic-memory integration

Measured on 2026-09-16 from the verified local MaleCNS v1.0 snapshot using seed 7.

## Provenance and scale

- Snapshot: `artifacts/male-cns-v1.0-w5`
- Dataset: `male-cns-v1.0-essential`
- Source manifest SHA-256: `8406eacdb75db4f1b84cfbf13b281b51628e68941458504312c0f42329fd5122`
- Snapshot SHA-256: `900eee63ac8d1e1805479a993735d41c20bdefb96e9c52d968a9db0b34709231`
- Graph: 166,606 neurons and 6,240,402 directed edges
- Plastic edges matched: 33,496; modified by the persisted state: 64
- Absolute plastic weight: 402,850.0 before and 402,708.999966383 after

## Protocol

The same deterministic cue schedule, Shiu parameters, seed, and initial state are used for each
baseline/learned pair. Cue spikes are emitted in batches of 8 every 23 steps, with 18-step
synaptic delay. The run lasts 430 steps (43 ms at `dt=0.1 ms`). The target MBON is `11402`.

## Results

| Condition | Baseline input | Learned input | Change |
|---|---:|---:|---:|
| Cue A (trained) | 258.50000762939453 | 219.72500228881836 | 15.0% decrease |
| Cue B (untrained control) | 253.82500457763672 | 253.82500457763672 | 0.0% drift |

Replay of the learned Cue A run was exact. The baseline graph digest was unchanged. Cue A target
spikes: 0 to 0; downstream spikes: 0 to 0; activity digest: unchanged. Cue B target and
downstream spikes and activity digest were also unchanged. The result passed all declared gates.

Runtime was 8.13 seconds with peak resident set size 964,788,224 bytes on this host. The complete
machine-readable result is `artifacts/shiu-plastic-memory-seed7.json`.

## Interpretation and limits

The changed conductance and voltage trajectories demonstrate that persisted plastic weights are
actually consumed by live sparse Shiu dynamics. They do not demonstrate intelligence, animal-like
learning, or useful behavior. An unchanged spike raster is a valid null result: subthreshold
plasticity can be real without changing the selected spike-count observables.

This stage still lacks sensory transduction, a physical body, muscles, proprioception, closed-loop
environmental feedback, validated neuromodulator dynamics, and behavioral holdout benchmarks. The
next scientific gate is therefore an embodied sensory-motor loop with explicit actions, feedback,
baselines, and out-of-sample evaluation.
