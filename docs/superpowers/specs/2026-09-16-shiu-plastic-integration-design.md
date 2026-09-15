# Shiu Plastic-Memory Integration Design

## Goal

Make the persistent MaleCNS KC→MBON memory causally affect the published Shiu whole-CNS event
dynamics. The phase must compare baseline and learned conditions under identical input events and
prove that any measured difference comes only from the persisted plastic multipliers.

This is a network-dynamics integration gate. It does not claim online reward learning, action
selection, embodiment, or intelligence.

## Chosen architecture

Load the immutable canonical snapshot into the existing sparse graph, validate a plastic-state
artifact against that exact snapshot, and materialize one copied event CSR whose matching KC→MBON
values are multiplied by the learned factors. The canonical Parquet files, baseline graph, edge
indices, topology, and signs remain unchanged.

This copy-and-patch design is preferable for the current phase because the event simulator keeps
its existing hot path and pays the mapping cost once. A dynamic multiplier lookup inside every
spike propagation would support online learning later but would add per-event overhead and a
mutable dependency to the core simulator. A separate plastic conductance channel would be useful
for receptor-aware neuromodulation, but it would change the published Shiu equations before the
necessary receptor and compartment evidence exists.

## Components and interfaces

### Snapshot identity

Move the canonical snapshot fingerprint into a small shared provenance module so extraction,
state loading, and event-graph integration use one definition. The SHA-256 stream covers, in this
order, each filename plus a NUL separator and the complete bytes of:

1. `metadata.json`
2. `source-annotations.parquet`
3. `edges.parquet`

The materializer rejects a state unless `dataset_id`, `source_manifest_sha256`, and
`snapshot_sha256` match the supplied snapshot. No best-effort or warning-only mode is allowed.

### Plastic event-graph materialization

The integration function accepts an `EventConnectome` and a loaded `PlasticEdgeSet`. It constructs
a neuron-ID-to-index map, resolves each `(pre_id, post_id)` pair into the outgoing CSR, and verifies:

- every plastic neuron ID exists;
- every plastic pair maps to exactly one explicit CSR entry after duplicate coalescing;
- the absolute canonical value equals the plastic baseline synapse count within float32 tolerance;
- the canonical sign is nonzero and is preserved;
- no topology, indices, or nonplastic value changes.

It returns a new `EventConnectome` with copied CSR storage. For a matched edge with canonical value
`w` and multiplier `m`, the learned value is `w * m`. The original graph is byte-stable at its
observable CSR arrays.

The materializer reports matched edges, modified edges, baseline/effective absolute weight, and
minimum/maximum multipliers. Dense neuron-by-neuron matrices are forbidden.

## Paired dynamic experiment

The experiment loads the existing association state and reuses its two recorded cue ensembles and
target MBON. Because the current NPZ state does not contain cue IDs, the CLI also accepts the
association metrics JSON and verifies that its state SHA-256 and snapshot identity match the NPZ.

For each cue, the protocol creates one deterministic schedule of direct voltage events targeting
the cue KCs. Baseline and learned conditions receive the exact same event arrays, seed, Shiu
parameters, duration, refractory exemptions, and initial state. In ascending neuron-ID order, cue
KCs are divided into batches of eight; one batch is driven every `refractory_steps + 1` steps with
the published Poisson voltage-jump amplitude. The run continues through the final batch, synaptic
delay, and five conductance time constants. This fixed stagger avoids one 64-neuron synchronous
pulse while changing neither published synaptic kinetics nor `synapse_mv`.

The runner records for each condition:

- total and per-role spike counts;
- target-MBON spike count;
- activity digest over step and neuron IDs;
- target-MBON voltage and conductance trajectories;
- integrated absolute target conductance;
- integrated cue-attributable delayed target input, accumulated from the exact propagation values
  generated when scheduled cue neurons spike;
- runtime and peak RSS.

Trajectory sampling is observational only and does not feed back into the simulation.

## Acceptance criteria

The integration passes only when all of the following hold:

1. Exactly all persisted plastic edges are matched and no nonplastic CSR entry changes.
2. Baseline graph arrays remain unchanged after materialization and both simulations.
3. Repeating a condition reproduces its activity digest and target trajectories exactly.
4. Learned cue A reduces integrated cue-attributable delayed target input relative to baseline by
   at least 10%, consistent with the accepted synaptic benchmark.
5. Cue B baseline-versus-learned cue-attributable target-input drift is at most `1e-6` relative.
6. The report states whether target or downstream spike activity changed; a spike difference is
   measured evidence, not a required result and not something the protocol may tune to obtain.
7. State, association metrics, and snapshot identity/checksum mismatches fail before simulation.

The cue-attributable metric is collected inside the live event loop from actual scheduled cue
spikes and graph propagation; it is not a static edge-sum substitute. Criterion 5 is evaluated on
cue-B edges, which retain multiplier 1.0. Whole-network cue-B activity and the complete target
conductance trajectory may still diverge if recurrent activity later reaches trained cue-A
pathways; the report separates direct target-input specificity from emergent recurrent effects.

## Failure handling and outputs

The CLI publishes one JSON result with the complete protocol, identities, parameters, resource
measurements, per-condition metrics, acceptance fields, and digests. It refuses overwrite and uses
staging plus atomic publication. Invalid or missing edges, nonfinite trajectories, zero baseline
denominators, malformed association JSON, checksum mismatch, and failed acceptance all return a
nonzero command result without publishing a partial artifact.

## Testing and real-data validation

Fixture tests cover identity mismatch, pair resolution, sign preservation, baseline immutability,
unmatched or wrong-weight edges, deterministic paired events, trajectory capture, acceptance, and
atomic CLI output. An opt-in real-data test runs against the local MaleCNS snapshot and persisted
association artifacts.

The measured real run must process the full 166,606-neuron, 6,240,402-edge event graph and all
33,496 KC→MBON plastic edges. Documentation records both positive and null results, including
whether the discrete spike raster changes.

## Next gate

After this phase, plastic weights affect real network dynamics but are still loaded between runs.
The next phase will update eligibility and dopamine during an ongoing simulation, derive dopamine
from an explicit environmental outcome, and connect opponent MBON activity to bounded motor/body
actions in a closed sensory–motor loop.
