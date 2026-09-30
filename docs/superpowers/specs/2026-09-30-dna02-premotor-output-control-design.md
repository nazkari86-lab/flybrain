# Locked design: ORN-driven DNa02 premotor-output control

## Biological question and evidence boundary

Does the literature-identified DNa02 → IN19A003/IN08A006 → leg-rotator
motor-neuron motif exert a *selective inhibitory output effect within the
retained MaleCNS/Shiu model* under registered ORN drive? This experiment
tests only the 33 signed inhibitory relay→motor edges. It does not establish
that DNa02 uniquely recruits the relays, that the motor cells drive a
particular muscle or stride, or that a living fly learns or steers. The
earlier strict DNa02 same-post control remains failed 0/6.

The route is defined by exact retained annotations before new outcome seeds:
the six ipsilateral IN19A003 and six ipsilateral IN08A006 cells, their 33
negative edges to `Sternal posterior rotator MN` and `Sternal anterior
rotator MN` respectively, and the 33 distinct motor posts. The snapshot
content SHA-256 must equal
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.
The code must verify all edge IDs, signs, types, sides, neuromeres, and counts
against the canonical snapshot and never mutate that graph.

## Candidate controls and frozen selection rule

For each of the 33 motor posts, collect all other negative canonical edges
from annotated `vnc_intrinsic` GABA cells to the *same post*, excluding the
12 route relays. A candidate source need not have the same cell type as the
route relay. Use intact food and threat ORN stimulation on **prior seeds
40–42 only** to count candidate and route-source spikes over 2,000 Shiu
steps per panel. For each edge form a six-vector of absolute signed model
weight × presynaptic spike count. Select one control edge per post by
minimizing the sum of absolute differences from the route-edge vector;
break ties by lower total absolute model-weight difference, then lower
source ID. Select a *distinct two-source pair* per post by the same
criterion applied to the sum of their exposure vectors; break ties by
lower total absolute model-weight difference, then lexicographic IDs.
Selection may use no new-seed outcomes. Record the selected IDs, positions,
weights, all six prior exposure vectors, and their selection hash before
running the new panel. Exposure is a rough simulator-input measure, not
physiological efficacy or matched time course.

## Prospective new-seed comparison

Use registered food and threat ORN populations from the hash-verified prior
artifact, `ShiuParameters(dt_ms=0.1, refractory_ms=2.0,
synaptic_delay_ms=1.0)` with all other defaults, 2,000 steps, and
previously unused seeds **43, 44, 45**. Generate one Poisson voltage-event
dictionary per odor/seed and reuse it for:

1. intact graph;
2. zeroing only the 33 route relay→motor edges;
3. zeroing 33 frozen single-control edges;
4. zeroing 66 frozen pair-control edges;
5. no-source (empty event dictionary, no edge change);
6. exact intact replay.

All edge interventions are sparse zero multipliers on an immutable graph.
Record full input-event, DNa02 and 12 relay spike-time digests, source and
target counts, 33 motor-target counts, aggregate other-`vnc_motor` counts,
whole-trace digest, actual input exposure in each intact panel, graph and
software hashes, configuration, runtime, and peak memory.

The **strict output-specificity gate** passes only if *every* one of the six
odor/seed panels satisfies all of the following, unchanged after outcomes:

1. Identical source-event digests across four source-driven conditions;
   DNa02, every one of the 12 relays, and the 33 pooled targets are recruited
   above no-source; no-source has zero DNa02, relay, and target spikes.
2. Identical DNa02 and relay spike-time digests across all four
   source-driven conditions. Aggregate spikes of other annotated motor
   cells are identical across these conditions.
3. Removing route edges increases pooled 33-target spikes by at least
   **25%** relative to intact and by at least **10 percentage points more**
   than either same-post control arm, using the same intact denominator.
4. On the intact panel, the two-edge control's summed absolute model
   weight×source-spike exposure lies within **80–120%** of the route's.
5. Intact replay is exact and the canonical graph digest is unchanged.

A failed recruitment, pairing, balance, margin, or replay check is a failed
gate, not a reason to adjust thresholds or reselect controls on seeds 43–45.
A positive gate would support model-level selectivity of these inhibitory
premotor outputs under the specified ORN input. It would not demonstrate
natural DNa02-dependent steering, measured inhibition in vivo, muscle
force, learning, or complete autonomous intelligence.

## Implementation and verification

Reuse the existing sparse graph, Shiu event generator, immutable overlay,
hash-verified ORN-source loader, and same-post condition readout. Keep
control selection separate from the outcome runner. Unit-test candidate
filtering, deterministic ties, exact graph-edge verification, and rejection
of malformed/future-seed controls. On the retained snapshot, validate the
route and controls with a short run before the locked six-panel outcome run.
Publish both positive and negative results with the protocol unchanged.
