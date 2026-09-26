# DNg33 direct-edge localization on a fixed neural source

## Question and locked anatomy

The [prospective embodied target-cell assay](../../data/dng33-direct-motor-targets-2026-09-27.md)
found a selective effect of silencing both DNg33 cells, but changed body
trajectories permit indirect and feedback explanations. Test whether the 16
retained direct DNg33→motor-target edges are necessary for a target response
under identical exogenous source events, without FlyGym or learning.

Source IDs are **13317, 13442**. Direct motor-target IDs, frozen before this
assay, are **800659, 803732, 810086, 813291, 814430, 814989, 815205,
815281**. The 16 positive edges have signed weight sum **404**.

The anatomical control is eight `vnc_intrinsic` postsynaptic cells connected
by both DNg33 sources: **802110, 802174, 803352, 805489, 802652, 802945,
904983, 805244**. These 16 positive edges have weight sum **379**, 6.19%
below the target-edge sum. The IDs were selected using one-to-one minimum
sum of absolute per-source weight differences among the 31 eligible
`vnc_intrinsic` cells; the matching is only a weight/edge-count control,
not a matched downstream-circuit or activity control. Fix these IDs and
weights before any outcome run. Do not retune either set after seeing data.

## Isolated neural protocol

Use the retained MaleCNS v1.0 weighted graph and the same Shiu parameters as
the embodied assay (`dt_ms=0.1`, `refractory_ms=2.0`,
`synaptic_delay_ms=1.0`, other defaults unchanged). For seeds **19, 20, 21**,
generate Poisson voltage events on the two DNg33 indices once per seed with
the published `poisson_voltage_events` function, 2,000 neural steps (200 ms).
Reuse the exact event array across these conditions:

1. `intact`: no edge overlay.
2. `direct_edges_zero`: set multipliers of exactly the 16 DNg33→motor-target
   CSR entries to zero via the existing sparse edge overlay.
3. `matched_edges_zero`: set multipliers of exactly the 16 DNg33→control
   `vnc_intrinsic` CSR entries to zero.
4. `no_source`: no voltage events and no overlay, to measure baseline.

Run `intact` again to check exact replay. Use fresh initial state for each
condition; no body, proprioception, visual feedback, plasticity, motor
decoder, rewards, planner, or RL. Do not modify canonical CSR arrays.
Resolve every biological ID to a CSR index and assert all 32 specified edges
exist, are positive, and retain the declared sums. Record the source-event
digest, per-condition DNg33 spike-time digest/count, all eight target counts,
the 700 other `vnc_motor` aggregate, and full neural-trace digest. Record
snapshot, source revision, graph digest before/after, and the exact removed
edge IDs/weights. The original `autonomous_behavior_claim_allowed` remains
false.

## Prospective gate and interpretation

A model-level **direct-edge necessity** signal requires in **every seed**:

1. `intact` has DNg33 and eight-target spikes, with eight-target total above
   `no_source`;
2. source-event digest and complete DNg33 spike-time digest are identical
   among `intact`, `direct_edges_zero`, and `matched_edges_zero`;
3. direct-edge removal reduces eight-target spikes by at least **20%** of
   intact, and its percent reduction exceeds the matched-edge control's
   percent reduction by at least **10 percentage points**;
4. intact replay is exact, all three interventions preserve the source
   revision/snapshot, and canonical graph digest is unchanged.

If any gate fails, report null/inconclusive without changing the seeds,
threshold, source drive, graph, targets, or control. Counts for each target
must be published even if the aggregate passes. A positive result establishes
edge necessity **in this model under this artificial input**, not exclusive
mediation of the earlier embodied effect, real synaptic physiology,
motor-neuron-to-muscle mapping, behavior, learning, or a living fly's
intelligence. Three simulator seeds are not a population confidence interval.
