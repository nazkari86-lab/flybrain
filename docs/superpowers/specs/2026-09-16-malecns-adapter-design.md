# MaleCNS v1.0 Streaming Adapter Design

## Objective

Convert the official MaleCNS v1.0 flat-connectome Feather exports into a canonical, provenance-rich sparse snapshot on a 16 GiB Apple Silicon host without loading the 151,856,684-row weight table into memory.

## Authoritative selection policy

The adapter reproduces the filtering mechanism in the official `flyconnectome/2025malecns` quantification notebook:

- retain annotated bodies whose `superclass` is non-empty and does not contain `tbc`;
- retain directed weights only when both endpoints are retained;
- use a configurable minimum connection weight, defaulting to the published `weight >= 5` analysis threshold;
- retain every selected neuron in the neuron table, including neurons isolated by thresholding;
- report both selected-neuron count and connected-neuron count instead of conflating them.

No expected neuron count is hard-coded as a filter. Counts are measured from the verified v1.0 artifacts and written to snapshot metadata.

## Canonical biological metadata

Each canonical neuron preserves body ID, type, superclass, side, source annotation status, inferred role, consensus neurotransmitter, and neurotransmitter provenance. Missing type, side, or neurotransmitter values remain explicit as `untyped`, `unknown`, or `unclear`.

Fast-transmitter sign policy:

- acetylcholine: `+1`, literature-level transmitter-sign assumption;
- GABA: `-1`, literature-level transmitter-sign assumption;
- histamine: `-1`, literature-level transmitter-sign assumption;
- glutamate: `0`, unresolved without receptor/circuit context;
- dopamine, octopamine, serotonin, and unclear: `0`, modulatory or unresolved.

Zero means “sign unresolved”, not “edge absent”. Canonical Parquet retains those rows. The current strict signed LIF runtime may omit zero-sign edges until the later receptor/neuromodulation engine is implemented; snapshot metrics must report how many synapses are unresolved.

## Streaming pipeline

1. Resolve four verified artifacts from the committed manifest and content-addressed cache.
2. Read the compact annotation and neurotransmitter tables.
3. Build an in-memory metadata map only for selected body IDs.
4. Write canonical neurons and preserved source annotations.
5. Iterate weight record batches, filter by endpoint membership and minimum weight, derive sign from the presynaptic transmitter, and append Parquet row groups.
6. Accumulate counts, total synaptic weight, unresolved-sign weight, connected body IDs, and peak RSS.
7. Atomically promote the completed snapshot and write metadata only after all source batches succeed.

The output contains `neurons.parquet`, `edges.parquet`, `source-annotations.parquet`, and `metadata.json`. A `.partial` directory is never treated as a valid snapshot.

## Failure behavior

- Missing or checksum-invalid artifacts stop before conversion.
- Duplicate annotation body IDs, duplicate neurotransmitter body IDs, invalid weights, and unknown weight endpoints are counted and reported; endpoint rows outside the selected population are filtered by policy, not treated as corruption.
- Output replacement is forbidden.
- Any exception leaves only a clearly named partial directory.
- Peak RSS above 8 GiB fails the real-data acceptance run.

## Testing and acceptance

Unit fixtures prove selection, metadata joins, sign policy, thresholding, batch-order invariance, atomic output, and exact metric accounting. The real v1.0 acceptance run must:

1. consume the four SHA-256-verified official files;
2. process every one of the 151,856,684 source weight rows;
3. produce a canonical snapshot without dense matrices;
4. report measured selected and connected neuron counts, retained edges, total retained weight, unresolved-sign fractions, runtime, and peak RSS;
5. pass schema validation and referential-integrity checks on the produced Parquet dataset;
6. remain above the 10 GiB free-space reserve.

This adapter does not claim that structural conversion alone produces intelligence. It is the required substrate for typed dynamics, sensory encoding, motor decoding, neuromodulation, and plasticity in subsequent phases.
