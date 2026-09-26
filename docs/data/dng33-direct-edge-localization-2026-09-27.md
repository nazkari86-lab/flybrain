# Direct DNg33 edges are necessary for target recruitment under fixed input in the MaleCNS model

The [embodied DNg33 target-cell assay](dng33-direct-motor-targets-2026-09-27.md)
showed a replicated closed-loop effect but could not separate direct synapses
from body feedback. A [prospectively locked isolated-neural assay](../superpowers/specs/2026-09-27-dng33-direct-edge-localization-design.md)
therefore compared zeroing the 16 direct DNg33→motor-target edges with zeroing
16 other DNg33 edges. Its [implementation](../../src/flybrain/dng33_edge_probe.py)
and tests were pushed as `842be22`; the completed source lock was pushed as
`454cfbdae10719cf6cfddb891cdcecefd54b129e` **before** outcome seeds
19–21. The predeclared gate passed in all three seeds.
The subsequent [annotation audit](dng33-abdominal-motor-identity-2026-09-27.md)
places all eight direct targets in A2–A5 and A7 with AbN-labelled exit nerves; this
is not an identified leg-muscle pathway.

## Fixed anatomy and intervention

The two DNg33 source IDs were 13317 and 13442. The eight frozen
`vnc_motor` targets were 800659, 803732, 810086, 813291, 814430, 814989,
815205, and 815281. Their 16 positive source edges have combined signed
graph weight +404. The control zeroed 16 positive edges from the **same two
sources** into eight nonmotor `vnc_intrinsic` cells (802110, 802174,
803352, 805489, 802652, 802945, 904983, 805244), with weight sum +379.
The control matches edge count, presynaptic identity, and approximate weight
(6.19% lower total), but **not** downstream connectivity or activity.

Each seed ran 2,000 Shiu neural steps (200 ms) on the same retained MaleCNS
graph with `dt_ms=0.1`, `refractory_ms=2.0`, and `synaptic_delay_ms=1.0`.
Poisson voltage events were generated once per seed for the two source cells
and reused exactly for intact, direct-edge-zero, and matched-edge-zero
conditions. The source cells were **not clamped**: observed source spike
times were checked and found exactly identical across these three
conditions. A no-source condition and intact replay were also run. Edge
zeroing used sparse multipliers, not changes to the canonical graph.
There was no FlyGym body, sensory feedback, plasticity, motor decoder,
reward, planner, or RL in this assay.

## Predeclared primary result

The locked gate required in *every* seed: source and target recruitment
above the no-source baseline; identical input-event and DNg33 spike-time
digests; at least 20% target reduction after direct-edge removal and at
least 10 percentage points more reduction than after matched-edge removal;
exact replay and unchanged canonical graph.

| Seed | Source voltage events | DNg33 spikes (identical across paired conditions) | Eight targets: intact → direct-edge zero → matched-edge zero | Other 700 `vnc_motor`: intact → direct-edge zero → matched-edge zero | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 19 | 70 | 102 | 71 → 0 → 75 | 495 → 495 → 447 | pass |
| 20 | 47 | 115 | 84 → 0 → 85 | 534 → 534 → 460 | pass |
| 21 | 56 | 109 | 78 → 0 → 80 | 501 → 501 → 443 | pass |

The direct-edge intervention removed **100%** of target spikes in each seed;
the matched intervention produced *more*, not fewer, target spikes (+5.63%,
+1.19%, +2.56% relative to intact). No-source target totals were zero.
All eight target cells had at least one spike in every intact seed, and
**every target count became zero** under direct-edge removal. Exact
per-cell counts for intact → direct-edge zero → matched-edge zero are:

| Target ID | Seed 19 | Seed 20 | Seed 21 |
| ---: | ---: | ---: | ---: |
| 800659 | 2 → 0 → 2 | 3 → 0 → 3 | 3 → 0 → 3 |
| 803732 | 12 → 0 → 13 | 14 → 0 → 14 | 13 → 0 → 14 |
| 810086 | 6 → 0 → 7 | 7 → 0 → 8 | 7 → 0 → 7 |
| 813291 | 4 → 0 → 4 | 5 → 0 → 5 | 5 → 0 → 5 |
| 814430 | 15 → 0 → 15 | 16 → 0 → 16 | 15 → 0 → 15 |
| 814989 | 7 → 0 → 8 | 9 → 0 → 9 | 8 → 0 → 8 |
| 815205 | 11 → 0 → 12 | 14 → 0 → 14 | 12 → 0 → 13 |
| 815281 | 14 → 0 → 14 | 16 → 0 → 16 | 15 → 0 → 15 |

Input-event and full DNg33 spike-time digests matched across each seed's
three source-driven conditions. The repeated intact condition matched the
original condition's complete recorded readout and neural-trace digest.
The graph digest remained unchanged, and all gates were independently
recomputed from the artifact rather than accepted solely from its Boolean
verdict. The matched-edge lesion lowered non-target motor spikes by 48, 74,
and 58, showing that this control intervention had downstream neural effects;
it was not an activity-matched control.

## What this does and does not establish

This is evidence that the **retained direct DNg33→target edges are necessary
for these eight target cells to spike under this artificial fixed-input
protocol in this Shiu/MaleCNS model**. Holding external input and observed
source spike timing equal eliminates changed body trajectories and changed
source firing as explanations *within this assay*. The result does not show
that the direct edges are the only path in other input regimes, or that they
exclusively mediate the earlier embodied lesion. Target response to these
external pulses cannot be equated with normal living-fly physiology, muscle
force, walking, choice, learning, or autonomous intelligence. Three seeds on
one connectome/parameterization are not a population confidence interval.
The project's `autonomous_behavior_claim_allowed` field remains false.

## Reproduction and artifact integrity

The output records clean source revision
`454cfbdae10719cf6cfddb891cdcecefd54b129e`, retained snapshot content
SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
166,606 graph neurons, 6,240,402 edges, all 32 resolved edge IDs/CSR
positions/weights, five conditions per seed, all gates, digests, and
per-target counts. Reproduce from that source revision and the official
MaleCNS products described in the [README](../../README.md):

```bash
.venv/bin/python -m flybrain.dng33_edge_probe artifacts/male-cns-v1.0-w5 \
  --output artifacts/repro-dng33-direct-edge.json \
  --steps 2000 --seeds 19 20 21
```

The [compressed JSON artifact](../../artifacts/dng33-direct-edge-localization-seeds19-21.json.gz)
was produced with `gzip -f -k -n -9` and passed `gzip -t`. SHA-256 of its
**uncompressed JSON bytes** is
`2d8d0260987700797eb5093dc83c17575c07770062c7c26dfefc09052b5b3613`.
Check with:

```bash
gzip -dc artifacts/dng33-direct-edge-localization-seeds19-21.json.gz | shasum -a 256
```
