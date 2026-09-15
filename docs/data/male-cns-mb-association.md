# MaleCNS Mushroom-Body Associative Memory

## Scope and provenance

This benchmark applies a persistent plastic overlay only to measured `Kenyon_Cell`→`MBON`
edges in the canonical MaleCNS v1.0 snapshot. It does not rewire anatomy. The result records the
dataset ID, source-manifest SHA-256, software commit, random seed, complete fitted parameters,
selected neuron IDs, state checksum, runtime, and memory use.

Measured anatomy:

| Population or edge set | Count |
|---|---:|
| Kenyon cells | 4,064 |
| dopamine neurons | 340 |
| MBONs | 97 |
| retained KC→MBON edges | 33,496 |
| retained KC→MBON synaptic weight | 402,850 |

The plasticity mechanism is a three-factor eligibility rule:

```text
e <- exp(-dt / tau_e) * e + pre_activity * post_gate
m <- clip(m - eta * dopamine * e, min_multiplier, max_multiplier)
```

Topology and baseline synapse counts are measured. Eligibility decay (`1000 ms`), learning rate
(`0.05`), multiplier bounds (`0.2` to `1.5`), cue size, and externally supplied dopamine are
explicit fitted assumptions. Positive dopamine depresses eligible KC→MBON weights.

## Measured real-topology run

Run on 2026-09-16 from software revision `5b27e87ac259678b986f02445f78e7544652a25d`:

```bash
/usr/bin/time -l uv run flybrain experiment mb-association \
  artifacts/male-cns-v1.0-w5 \
  --seed 7 --cue-size 64 --trials 3 \
  --state-output artifacts/mb-association-seed7-state.npz \
  --output artifacts/mb-association-seed7.json
```

The benchmark selected MBON `11402`, which has the largest number of distinct retained KC inputs
(`1,502`). Seed 7 selected two disjoint 64-KC ensembles connected to that MBON.

| Condition | Before | After | Relative change |
|---|---:|---:|---:|
| cue A paired with dopamine, 3 trials | 940.0 | 798.999969959259 | 15.000003% decrease |
| untrained cue B | 923.0 | 923.0 | 0 |
| cue A, dopamine absent, 3 trials | 940.0 | 940.0 | 0 |
| cue A, eligibility cleared before dopamine, 3 trials | 940.0 | 940.0 | 0 |

| Resource or persistence metric | Measured value |
|---|---:|
| benchmark runtime | 1.64 seconds |
| independent maximum RSS | 196,788,224 bytes |
| plastic-state edges reloaded | 33,496 |
| exact effective-weight replay | true |
| acceptance gate | passed |

Canonical snapshot SHA-256 (metadata, source annotations, and canonical edges):

```text
900eee63ac8d1e1805479a993735d41c20bdefb96e9c52d968a9db0b34709231
```

Plastic-state SHA-256:

```text
6c69acecad693958ee4514ecb94d3bb66d9c017d7f373f984eea9f3137b54169
```

The state uses format version 1 and embeds dataset ID, source-manifest SHA-256, and canonical
snapshot SHA-256. Loading rejects non-finite, malformed, or identity-incomplete state.

A fresh run with the same seed reproduced the selected MBON, both cue-ID lists, every response,
all relative changes, parameters, and exact persistence replay. Runtime, peak RSS, and output paths
are intentionally outside the deterministic contract.

## Interpretation and limitations

This demonstrates persistent, cue-specific, dopamine-gated synaptic memory on real MaleCNS
KC→MBON topology, with timing and no-dopamine controls. It is not evidence of embodied
intelligence or learned behavior. The response is currently the summed effective input weight to
one MBON, dopamine is imposed by the benchmark rather than produced by an outcome-sensitive
network, and the plastic overlay is not yet integrated into the live Shiu event simulation.

The next validity gates are to integrate effective weights into whole-CNS event propagation,
derive dopamine from world outcomes, add opponent MBON/action selection, and close the loop through
sensory receptors, body dynamics, proprioception, and motor feedback.
