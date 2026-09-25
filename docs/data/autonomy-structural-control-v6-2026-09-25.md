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

The artifacts were generated from a dirty working tree before the source commit;
their `software_revision` field says so. The SHA-256 values identify the local
files, and the committed source plus tests identify the code revision. A clean-
revision retained artifact remains a reproducibility task.

## Next scientific gate

Before a positive learning claim, run a preregistered v6 protocol with independent
seeds and a physically adequate horizon; require normal to outperform
no-plasticity, DAN lesion, KC→MBON lesion and structural rewiring for both food
seeking and threat avoidance. The present negative effects point to the need for
pathway-specific MBON→DN→motor and locomotion validation, not a looser claim gate
or hidden planner. Full biological intelligence would additionally require
independent physiological and behavior comparisons, broader senses and internal
state, and physical transfer beyond this reference body.
