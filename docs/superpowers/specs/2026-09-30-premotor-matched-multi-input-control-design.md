# Locked design: activity-matched multi-input premotor control

## Question and boundary

Does removing the 33 IN19A003/IN08A006 inhibitory outputs produce a larger
response in their 33 annotated leg-rotator motor posts than removing alternative
GABAergic input to those **same posts**, when the alternative input is actually
matched in modeled presynaptic exposure? This repairs the under-exposure of the
published one- and two-edge controls; it does not reclassify their failed
43–45 gate. It remains a MaleCNS/Shiu model test, not living-fly physiology,
natural steering, muscle force, or learned autonomy.

## Frozen inputs and selection

Reuse the exact snapshot SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
the existing registered food/threat ORN source artifact, 2,000 Shiu steps,
and parameters `dt_ms=0.1`, `refractory_ms=2`, `synaptic_delay_ms=1` with
all other defaults. Reuse the 33 type- and anatomy-checked route edges and
the existing same-post candidate filter. The earlier 43–45 outcomes may not
enter control selection or change any threshold.

Count intact presynaptic spikes in six **prior** panels: food and threat,
seeds 40, 41, 42. For each motor post, enumerate distinct subsets of one to
four non-route candidate edges. Choose the subset minimizing the sum of six
absolute differences between candidate and route `abs(weight) * spikes`
vectors. Break ties by smaller absolute weight-sum difference, then fewer
edges, then lexicographically smaller source-ID tuple. No candidate may be
reused for the same post. Persist exact edge IDs, CSR positions, weights,
per-panel exposures, event digests, graph and source hashes, and software
revision in a frozen control artifact. The code must reject any changed
identity, duplicated position, or malformed control map.

Before looking at new outcomes, calculate the pooled alternative/route
exposure ratio for each prior panel. If any is outside **0.8–1.2**, report
control feasibility failure and do not run the new holdout. This threshold
is retained from the first protocol and is not tuned after the fact.

## Prospective holdout and gates

Only after source and control selection are committed, run the previously
unused seeds **46, 47, 48** for food and threat. Generate one ORN-event
dictionary per panel and reuse it for intact, route-zero, selected-control-
zero, no-source, and exact intact replay. Sparse zero multipliers must leave
the canonical graph unchanged. Capture DNa02, relay and alternative-source
spike-time digests, 33 post counts, other-motor counts, input exposure,
runtime, peak memory, and output hashes.

For each of six panels, require all of:

1. Identical ORN events, DNa02 spike times, and all monitored presynaptic
   spike times across source-driven arms; exact replay and unchanged graph.
2. No-source quiet; DNa02, every one of 12 relays, and pooled motor targets
   recruited above no-source. Other annotated motor spikes stay identical.
3. Alternative/route pooled exposure ratio in **0.8–1.2** on the intact arm.
4. Route-zero increases pooled motor spikes at least **25%**, and at least
   **10 percentage points** more than the matched control-zero arm, both
   relative to intact.

The overall specificity gate passes only if all six panels pass. Report
individual failures and raw values even when it does not pass. A positive
result supports only comparative inhibitory-output influence *in this model*;
it does not establish that DNa02 uniquely recruits the relays, prove measured
inhibition, or imply a whole-animal behavioral effect.

## Implementation and verification

Use a separate versioned runner so the published protocol and artifacts are
immutable. Reuse the current sparse graph, source loader, event generator,
condition readout, and route annotation checks. Unit-test deterministic
selection, tamper rejection, input/source pairing, and gate arithmetic.
Record exact software and data provenance. Run the routine Ruff, mypy and
pytest gates, then independently recompute the reported numeric gate from
the holdout JSON. Publish positive and negative outcomes without adjusting
thresholds or control identities after seeing new seeds.
