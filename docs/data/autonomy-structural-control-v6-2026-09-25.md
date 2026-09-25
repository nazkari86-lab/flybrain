# Autonomous behavior structural-control audit — 2026-09-25

The objective is still learned, autonomous, biologically grounded fly behavior.
These runs do **not** establish it. The retained MaleCNS model remains a simulation
with assumed cellular dynamics, odor field, motor decoder and body parameters.

## What changed

- The sparse KC→MBON multiplier is now applied during event propagation instead of
  copying the entire 6,240,402-edge CSR graph for every 10 ms neural window.
  The canonical graph remains immutable. Synthetic tests compare the new path
  exactly with a copied effective graph for both small and 70-neuron spike batches,
  including multiplier updates and zero weights.
- The v6 `rewired_control` swaps actual KC→MBON targets on an isolated graph copy.
  It preserves each source's outdegree and each target's indegree, rejects duplicate
  edges, records the resulting graph digest and changed-edge count, and requires
  at least 50% of plastic edges to change in every replicate before a behavioral
  claim can pass. If no valid swap exists, this gate fails closed.
- The benchmark envelope is v2 and its evidence protocol is
  `measured-replay-persistent-memory-v6`. Pre-v6 artifacts had a shuffled
  plasticity-routing control, **not** a structurally rewired graph; they cannot
  support a claim of superiority to structural rewiring.

## Measurements

On the retained 166,606-neuron, 6,240,402-edge graph, a synthetic 1,000-source
spike batch with 33,496 overlay locations took 1.51 ms per call through the new
propagator, versus 8.74 ms to copy and multiply the full CSR array (10 calls each,
one local machine). This is an operation-level comparison, not a demonstrated
whole-benchmark speedup.

The pre-v6 20-step single-seed diagnostic is
`artifacts/autonomous-behavior-current-overlay-steps20.json`
(SHA-256 `7c7bbff04d897d9ecde4de17b6f75c7a7d1210893b94e043766e54fe7d3ef971`).
Normal minus no-plasticity showed -394 mean MBON spikes, +7.75 descending spikes,
-6.25 motor spikes, food score -0.000238 and threat score -0.000840. These are
paired count and score differences, **not** proof that the MBON difference caused
the downstream changes. The result has only one independent seed, 20 body steps,
and no structurally rewired control, so it cannot support a behavioral claim.

The v6 two-step structural-control smoke artifact is
`artifacts/autonomous-behavior-v6-structural-rewire-seed7-steps2.json`
(SHA-256 `5eee6837e73726a185e6aa2bfbfcd983fc05d4bc59e34ddbd64ad5222013898a`).
Its isolated control changed 30,964 of 33,496 declared plastic edges (92.44%).
Normal observations, neural summaries and training summaries had identical
canonical-JSON hashes to the prior two-step run, confirming that building the
control did not alter the normal condition. All 30 episodes replayed exactly;
all 20 holdout weights and memories
were frozen, and the canonical graph was unchanged. All food and threat deltas
were zero; `behavioral_claim_allowed=false`. Two steps and one seed are only a
protocol smoke test.

The two artifacts above were generated from a dirty working tree before the
source commit; their `software_revision` field says so. A fresh run from clean
source revision `04cd81c66bd4f72ef9f7932a33615e478f2ab20a` is published as
[`artifacts/autonomous-behavior-v6-clean-04cd81c-seed7-steps2.json`](../../artifacts/autonomous-behavior-v6-clean-04cd81c-seed7-steps2.json)
(SHA-256 `2010f71654614b64403997d9a67276eba7c8e97e58ead6b3d9e35e2065cc59e4`).
Its `software_revision` is that exact commit without a dirty suffix. The normal
observations, neural summaries and training summaries match the earlier v6 smoke
artifact byte-for-byte after canonical JSON normalization. It again rewires
30,964/33,496 edges, replays exactly, freezes holdout weights and memory, and
returns zero food and threat differences with `behavioral_claim_allowed=false`.
This artifact establishes reproducibility of the short protocol run, not learning.

A clean-revision 20-step diagnostic is published as
[`artifacts/autonomous-behavior-v6-diagnostic-04cd81c-seed7-steps20.json`](../../artifacts/autonomous-behavior-v6-diagnostic-04cd81c-seed7-steps20.json)
(SHA-256 `aa6b626df0dbab35c71b3d5f08ea4727927580c66e58715d4806060ce17d56e9`).
Its `software_revision` is `3751973a34b6b29dce5bbb1feb14ae7f673d01a9`.
Canonical-JSON hashes of normal observations, neural activity and training
summaries match the pre-v6 20-step diagnostic, as do comparisons to no
plasticity, DAN lesion and KC→MBON lesion. The new structural control rewired
30,964/33,496 plastic edges. Normal-minus-no-plasticity food and threat scores
were -0.000238 and -0.000840; normal-minus-structural-control scores were
-0.000553 and -0.000125. All 30 episodes replayed exactly, holdout plasticity
and memory were frozen, and the canonical graph was unchanged. These are
negative, single-seed diagnostic effects at 20 steps, below the prespecified
100-step minimum; they do not support learned autonomous behavior.

The same one-seed, 20-step v6 protocol was then run on the millimetre-scale
FlyGym 2.1.0 / MuJoCo 3.9.0 body. The published artifact is
[`artifacts/autonomous-behavior-v6-flygym-b2b046d-seed7-steps20.json`](../../artifacts/autonomous-behavior-v6-flygym-b2b046d-seed7-steps20.json)
(SHA-256 `59d919e86f6a7917927fb7fcd6f7f73b30dc91fd645afb3e4ffe3d3a211b78be`,
clean source revision `b2b046de49e55c3044d315ab25b3f95c66ad676d`).
Its structural control again rewired 30,964/33,496 plastic edges; all 30
episodes replayed exactly, holdout weights and memory were frozen, the graph was
unchanged, and every normal holdout ended physically supported and unfallen.
There were no food or threat contacts. Against no plasticity, normal food score
was lower by 0.00000916 m while threat score was higher by 0.00002139 m. The
food and threat differences against structural rewiring were -0.00001732 m and
+0.00003453 m. These are millimetre-scale displacement-score differences from
one seed, not independent confidence intervals; their mixed signs and the
subminimum horizon keep `behavioral_claim_allowed=false`. The more realistic
body therefore does not rescue the current food-and-threat learning claim.

The motor interface is a specific unclosed biological boundary: direct
inspection of the configured FlyGym `LEGS_ONLY` model found 66 leg joint DOFs,
of which this neural adapter actuates 18 (three per leg), with no adhesion
actuators. The separately labelled engineering baseline uses FlyGym's official
42-DOF plus six-adhesion hybrid controller and moved 3.36 mm in 0.32 s in its
own run; this is not a matched intervention or evidence that substituting that
controller would make the connectome intelligent. The retained annotations
contain additional named motor classes, but they do not yet establish a
complete, validated neuron-to-muscle map for all six legs. Next, test
cell-specific motor recruitment and physical joint effects before expanding
the neural actuator bridge; do not silently insert the official gait policy
into the biological path.

## Next scientific gate

Before a positive learning claim, run a preregistered v6 protocol with independent
seeds and a physically adequate horizon; require normal to outperform
no-plasticity, DAN lesion, KC→MBON lesion and structural rewiring for both food
seeking and threat avoidance. The present negative effects point to the need for
pathway-specific MBON→DN→motor and locomotion validation, not a looser claim gate
or hidden planner. Full biological intelligence would additionally require
independent physiological and behavior comparisons, broader senses and internal
state, and physical transfer beyond this reference body.
