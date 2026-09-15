# FlyBrain Platform Design

## Purpose

Build a reproducible, closed-loop digital fruit-fly nervous-system research platform that uses published connectome structure as the controller. The system must separate measured biological data, explicit modelling assumptions, fitted parameters, and observed simulation results. Game score is a downstream benchmark, not evidence of biological fidelity.

## Success criteria

The platform is successful only when it can:

1. Reproduce selected published sensorimotor interventions on a versioned connectome model.
2. Run a sensory-to-motor closed loop without a hidden policy network or hard-coded task solver.
3. Learn at least one new association through localized biologically motivated plasticity.
4. Quantify causal effects through repeatable neuron, cell-type, and synapse lesions.
5. Report runtime, peak memory, dataset provenance, assumptions, and uncertainty for every experiment.

## Scientific boundaries

- A connectome is structural evidence, not a complete measurement of membrane, receptor, synaptic, hormonal, developmental, or individual-state parameters.
- The simulator will not claim to recreate the scanned animal or consciousness.
- Published outcomes will be labelled `reproduced` only after matching a predeclared metric and tolerance. Otherwise they remain `attempted` or `exploratory`.
- Reinforcement learning may tune uncertain model parameters or provide a separately labelled baseline. It may not generate the fly's actions in the primary system.
- Every inferred sign, conductance, delay, receptor response, and plasticity coefficient must carry provenance and confidence metadata.

## Scope decomposition

The complete platform consists of six independently testable subprojects. Each receives its own implementation plan and acceptance gate.

1. **Connectome Core:** provenance-aware data acquisition, canonical sparse graph, neuron/synapse metadata, event-driven LIF reference simulator, deterministic experiment runner.
2. **Biological Dynamics:** typed LIF/AdEx populations, transmitter and receptor rules, morphology-derived delays where available, neuromodulation, localized plasticity.
3. **Validation Suite:** feeding, grooming, looming/escape, locomotion, and turning experiments with lesions and published comparators.
4. **Embodied Fly:** FlyGym/NeuroMechFly adapter, retina/ommatidia, olfaction, touch, wind, proprioception, muscles, and closed-loop state feedback.
5. **Autonomous Curriculum:** food seeking, threat avoidance, target following, Pong, Flappy Bird, maze, and Doom navigation without privileged game state.
6. **Causal Laboratory:** activity inspection, counterfactual stimulation/silencing, before/after-learning ablations, cell-type importance maps, and exportable experiment bundles.

Only Connectome Core is in the first implementation plan. This prevents an unvalidated monolith while preserving stable interfaces for later stages.

## Architecture

```text
versioned datasets -> canonical importer -> sparse connectome snapshot
                                              |
sensory adapters -> encoded spike events -> dynamics engine -> motor readout
       ^                                      |                 |
       |                                      v                 v
world/body <- proprioceptive feedback <- activity recorder <- effectors
                                              |
                                      lesion/plasticity hooks
```

### Dataset registry

A machine-readable manifest records source URL, source publication, licence, expected size, immutable checksum, retrieval date, schema adapter, and local status. Downloads are resumable and content-addressed. Raw source files are never silently rewritten. Derived files include a manifest linking them to exact inputs and transformation code.

Because the host currently has about 32 GiB free, acquisition uses a storage budget. Metadata and compact edge tables are prioritized; meshes, imagery, and duplicate exports are optional. The command must estimate required space before downloading and refuse when the reserve would fall below 10 GiB.

### Canonical connectome

The internal snapshot stores neurons and directed synaptic edges in columnar files. Required neuron fields are stable ID, source dataset, cell type, hemisphere/side, anatomical regions, transmitter annotation, sensory/motor role, and confidence. Required edge fields are pre/post IDs, synapse count, sign provenance, confidence, optional location/path length, and source record.

No dense neuron-by-neuron matrix is permitted. Runtime representations use compressed sparse arrays or event adjacency lists. FlyWire/FAFB and MaleCNS adapters map into the same versioned schema without erasing source-specific fields.

### Dynamics engine

The reference engine is deterministic and CPU-first. It begins with event-driven LIF dynamics because this is testable against published whole-connectome work. Model classes are replaceable per cell type, allowing AdEx and richer compartmental models later. Synaptic efficacy is calculated from observed synapse count plus explicit transmitter, receptor, reliability, and fitted-scale terms; each term remains inspectable.

Randomness is supplied by named seeded streams. Checkpoints include simulation state, plastic weights, dataset IDs, configuration hash, software revision, and seed.

### Experiment protocol

Experiments are declarative configurations containing input population, stimulus timing, interventions, measured outputs, comparator, metric, tolerance, duration, and seed set. Outputs include trajectories, selected spike/activity traces, aggregate rates, intervention effect sizes, runtime, and peak memory.

The first validation targets are small synthetic circuits and one published sensorimotor pathway before attempting a full connectome run. Full-scale execution is blocked until importer integrity, sign accounting, determinism, and resource guards pass.

## Error handling and safety

- Checksum, schema, referential-integrity, duplicate-ID, invalid-sign, and disk-budget failures stop the run with actionable diagnostics.
- Unknown biological parameters are represented explicitly; they are never replaced by undocumented defaults.
- Partial downloads remain quarantined and cannot enter experiments.
- Every long run writes an atomic checkpoint and an append-only event log.
- External source disappearance does not invalidate an existing verified content-addressed artifact.

## Testing strategy

Development follows test-first red/green/refactor cycles.

- Unit tests cover manifest validation, checksum handling, schema conversion, sparse graph invariants, dynamics equations, delays, seeded noise, lesions, and checkpoint restoration.
- Property tests cover edge-order invariance, no creation of nonexistent neurons, conservation of event ordering, and deterministic replay.
- Golden tests use tiny hand-calculated circuits with analytically predictable spikes.
- Dataset contract tests run on committed miniature fixtures, never requiring network access.
- Integration tests run a small sensory-to-motor experiment and compare declared metrics.
- Benchmarks fail when memory complexity approaches dense `O(N^2)` behavior.

## Technology choices

- Python 3.12+ for orchestration and scientific interoperability.
- NumPy, SciPy sparse arrays, PyArrow/Parquet, Pydantic, pytest, and Hypothesis in the initial core.
- Optional Numba/JAX/Rust acceleration is admitted only after profiling identifies a bottleneck and parity tests prove equivalent dynamics.
- `uv` manages locked environments when available; a standard `pyproject.toml` remains the source of truth.

## First milestone acceptance gate

Connectome Core is accepted when a clean checkout can:

1. Validate a source manifest and safely acquire a small fixture.
2. Convert it to the canonical sparse snapshot with provenance retained.
3. Run a seeded LIF sensory-to-motor circuit and reproduce its expected trace.
4. Silence a declared population and report the causal output difference.
5. Replay from a checkpoint byte-for-byte at the declared observable layer.
6. Pass tests, static checks, and a memory-scaling benchmark.
7. Produce an experiment bundle containing configuration, metrics, provenance, logs, and environment metadata.

## Deferred work

MaleCNS full-data ingestion, morphology-derived delays, FlyGym embodiment, biological learning, visualization, and game adapters are intentionally deferred to later subproject specifications. Their interfaces are anticipated here, but no unvalidated implementation is bundled into the first milestone.
