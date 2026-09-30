# DNa02 afferent-isolated thoracic edge assay: prospective design

## Question and boundary

In the retained MaleCNS/Shiu model, do the **same 17 direct DNa02→registered
thoracic motor edges** account for more target spikes than a weight-matched
17-edge intervention from the same sources when DNa02's source spike timing is
held equal across conditions? This tests a causal model circuit under
artificial source drive. It does not test a natural stimulus, muscle force,
walking, learning, or an animal. The earlier unisolated assay is preserved as a
failed primary gate: its matched-edge intervention changed DNa02 firing.

## Fixed source, anatomy, and controls

- Use the retained snapshot with content SHA-256
  `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.
  Keep the canonical 166,606-neuron/6,240,402-edge CSR unchanged.
- DNa02 biological IDs are `10360` and `523769`.
- Reuse **exactly** the 17 target and 17 `vnc_intrinsic` control edges, their
  source/post IDs, positions, and weights from the already published
  `artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz`. Its uncompressed
  JSON SHA-256 is
  `3d2dc99233197e45d2849aa6b52220689e2b230fcb53f85bce991ac1ba8b0dc5`.
  Target weight sum must be 874, control sum 887. Do not reselect edges after
  seeing any outcome.
- Find every canonical CSR position whose postsynaptic index equals one of
  the two DNa02 source indices. The current read-only preflight found 1,136
  such incoming edges (579 to `10360`, 557 to `523769`), with no self-edges.
  Zero this **same** complete afferent set in *every* condition via a sorted
  sparse overlay. A mismatch in count, IDs, overlap, or snapshot hash aborts
  the assay before any outcome is accepted.

The common afferent lesion is necessary to hold the two source spike trains
against recurrent feedback; it is a substantial artificial manipulation and
cannot be described as intact physiology. It is not part of the 17-edge
intervention whose effect is compared. The rest of the graph remains present.

## Locked panel

For each of the previously unused seeds **28, 29, 30**, run 2,000 Shiu steps
(200 ms) with `dt_ms=0.1`, `refractory_ms=2.0`,
`synaptic_delay_ms=1.0`, and all other `ShiuParameters` defaults. Generate
Poisson voltage events once per seed for the two DNa02 sources and reuse
the exact events in all source-driven conditions. No plasticity, body,
sensory feedback, motor decoder, reward, or RL is present.

1. `intact`: common source-afferent zero overlay only.
2. `direct_edges_zero`: common overlay plus the 17 frozen DNa02→target edges.
3. `matched_edges_zero`: common overlay plus the 17 frozen DNa02→intrinsic
   control edges.
4. `no_source`: common overlay but no source voltage events.
5. `replay`: exact repeat of `intact`.

The primary gate passes only if **every** seed has: positive DNa02 spikes and
target total above `no_source`; identical source-event and observed source-
spike-time digests across the first three conditions; at least 20% fewer
target spikes after direct-edge zeroing; a direct reduction at least 10
percentage points larger than matched-edge reduction; exact intact replay;
and unchanged canonical graph digest. Record all per-cell target counts,
other annotated motor-cell aggregate counts, source spikes/digests,
condition trace digests, edge identities, overlay count, parameters, seed,
software revision, and snapshot digest. A failed gate remains failed; do not
change thresholds, controls, horizon, or seeds in response.

## Interpretation

If the gate passes, the bounded statement is that these anatomically
thoracic direct connections are intervention-specific contributors to their
target-cell activity **in an afferent-isolated Shiu/MaleCNS model under
artificial DNa02 input**. It does not rescue the earlier embodied gate or
establish a leg-muscle route, realistic electrophysiology, learned behavior,
or autonomous intelligence. If it fails, report which invariant failed and
retain the negative artifact. A technical success alone is not a biological
behavior result.
