# Evidence-first autonomous learning implementation plan

**Goal:** Add biologically bounded KC-to-MBON/DAN associative learning and high-fidelity physics
parity without introducing a hidden policy or privileged task state.

**Dependency:** Complete Tasks 6–10 in
`docs/superpowers/plans/2026-09-17-hexapod-motor.md` first. A retained sensorimotor result may be
null or underpowered, but its infrastructure, integrity, replay, and artifact gates must pass before
learning is connected to behavior.

**Spec:** `docs/superpowers/specs/2026-09-17-autonomous-learning-biological-realism-design.md`

## Global constraints

- Canonical graph arrays, topology, signs, and nonplastic weights remain immutable.
- Only exact existing KC-to-MBON edges may receive a sparse bounded multiplier.
- DAN sign-zero adjacency is modulatory routing only, never ordinary voltage propagation.
- No scalar reward, target coordinate, desired action, planner, PPO, CNN, LLM, or direct body
  command may reach the primary agent.
- All selectors and edge manifests bind exact IDs, counts, hashes, retained artifacts, and evidence.
- Every feature follows RED–GREEN–REFACTOR, focused gates, full pytest/Ruff/mypy, retained-data
  verification, and immutable artifact audit.
- Real null, directionally wrong, or underpowered results are retained without post-hoc retuning.

### Task 1: Register exact KC, MBON, and DAN populations

**Files:** create `data/registry/autonomous-learning-registry-v1.json`, create
`tests/test_autonomous_learning_registry.py`, extend `src/flybrain/biological_registry.py` only if
strict edge-bound declarations require a new typed model.

1. Write failing tests for complete `class=Kenyon_Cell`, `class=MBON`, and `class=DAN` populations,
   exact sides/types, expected counts, ID hashes, overlap, graph membership, and snapshot drift.
2. Add literature-backed appetitive and aversive DAN type declarations. Unknown-valence DAN types
   remain registered but unusable as reinforcement inputs.
3. Resolve against the retained MaleCNS artifact and record exact IDs/hashes without truncation.
4. Run focused and full schema gates; commit `data: register MaleCNS learning populations`.

### Task 2: Bind the exact plastic edge manifest

**Files:** create `src/flybrain/plastic_edge_registry.py`, create
`tests/test_plastic_edge_registry.py`, extend the learning registry artifact.

1. Write failing tests for exact existing KC-to-MBON edges, edge count, contact sum, pre/post
   coverage, edge digest, sign, source identity, and deterministic ordering.
2. Resolve only canonical positive KC-to-MBON edges; reject additions, duplicate pairs, missing
   neurons, sign drift, weight drift, and dense materialization.
3. Register DAN-to-KC and DAN-to-MBON sign-zero adjacency as a distinct modulatory manifest.
4. Verify retained measurements and commit `data: bind MaleCNS plastic edge manifest`.

### Task 3: Implement the immutable sparse plastic overlay

**Files:** create `src/flybrain/plastic_overlay.py`, create `tests/test_plastic_overlay.py`.

1. Write failing tests for unit initialization, bounded sign-preserving multipliers, sparse storage,
   reset, serialization, digest stability, graph immutability, and exact replay.
2. Store only canonical edge indices and multipliers; expose sparse event propagation without
   rewriting CSR data.
3. Reject unknown edges, nonfinite updates, sign reversal, out-of-range multipliers, aliasing, and
   topology drift.
4. Run focused/full gates and commit `feat: add sparse KC-MBON plastic overlay`.

### Task 4: Add local eligibility and DAN modulation

**Files:** create `src/flybrain/mushroom_body_learning.py`, create
`tests/test_mushroom_body_learning.py`.

1. Define frozen learning parameters, state, update records, and restoration records.
2. Write failing tests proving KC/MBON activity creates only local traces, traces decay, DAN alone
   changes nothing, eligibility without DAN changes nothing, matching DAN changes only routed
   edges, and sign-zero DAN edges never inject voltage.
3. Implement bounded depression and recovery with exact sparse ordering and deterministic replay.
4. Add no-plasticity, DAN lesion, MBON lesion, edge lesion, restoration, perturbation, and holdout
   tests; commit `feat: implement local DAN-gated KC-MBON plasticity`.

### Task 5: Build nonprivileged reinforcement interfaces

**Files:** create `src/flybrain/reinforcement_interface.py`, create
`tests/test_reinforcement_interface.py`.

1. Define anonymous odor/visual/contact observations and exact sensory-to-DAN calibration records.
2. Prove encoders expose no object identity, target, distance, reward, heading, or desired action.
3. Keep direct DAN stimulation in a separate calibration API and never reuse it as the full-path
   sensory condition.
4. Add mirror, monotonicity, bank normalization, timing, lesion, and replay tests; commit
   `feat: add biological reinforcement interfaces`.

### Task 6: Create conditioning environments and schedules

**Files:** create `src/flybrain/conditioning_world.py`, create
`tests/test_conditioning_world.py`.

1. Build deterministic appetitive and aversive train/test episodes with anonymous cues, randomized
   layouts, collision physics, and prebuilt hashed schedules.
2. Expose only retinal, olfactory, tactile, and proprioceptive observations to the agent.
3. Generate unseen-cue, unseen-layout, mass, friction, delay, and damaged-leg holdouts before the
   retained run.
4. Prove schedule independence from neural outputs and commit `feat: add conditioning worlds`.

### Task 7: Run the autonomous-learning causal benchmark

**Files:** create `src/flybrain/autonomous_learning_benchmark.py`, create
`tests/test_autonomous_learning_benchmark.py`.

1. Integrate sensory events, Shiu state, plastic overlay, DN/motor decoding, reference-body steps,
   and delayed proprioception without resetting state inside an episode.
2. Classify plasticity calibration separately from appetitive and aversive behavior.
3. Add matching/opposite lesions, no-plasticity, restoration, replay, fair random controls,
   degree-preserving rewired ensembles, shuffled overlays, perturbations, and holdouts.
4. Predeclare activity, preference, lesion, gait, energy, stability, and generalization thresholds.
5. Use known-positive and known-null synthetic circuits; never force a retained positive outcome.
6. Commit `feat: benchmark autonomous associative learning`.

### Task 8: Add FlyGym/MuJoCo backend parity

**Files:** create `src/flybrain/hexapod_backend.py`, create `src/flybrain/flygym_backend.py`, create
`tests/test_hexapod_backend.py`, create `tests/test_flygym_backend.py`.

1. Extract the reference torque/observation protocol behind a strict backend interface.
2. Install a pinned compatible FlyGym/MuJoCo environment and record dependency/platform identity.
3. Map identical bounded torque and observation records without backend-specific policy inputs.
4. Test joint conventions, units, contacts, mirror, replay tolerance, interventions, and learned
   effect direction without retraining.
5. Treat unavailable or incompatible FlyGym as a failed optional parity gate, not a reason to alter
   the reference result; commit `feat: add FlyGym parity backend`.

### Task 9: Publish immutable learning artifacts

**Files:** modify `src/flybrain/cli.py`, create `tests/test_autonomous_learning_cli.py`, create
`docs/data/male-cns-autonomous-learning.md`, update `README.md`.

1. Add an atomic CLI containing registry, graph, plastic manifest, initial/final overlay, schedules,
   interventions, classifications, software revision, runtime, and peak RSS.
2. Refuse overwrite, aliases, malformed evidence, graph drift, partial restoration, and partial
   output.
3. Run fixture and retained MaleCNS experiments and preserve every null/underpowered result.
4. Audit artifact SHA-256 and commit `docs: validate MaleCNS autonomous learning`.

### Task 10: Whole-branch scientific review

1. Audit population and edge selectors, DAN valence evidence, modulatory routing, plastic locality,
   reward leakage, body-command leakage, graph mutation, random-control fairness, threshold
   preselection, dense allocation, reset/replay, and atomicity.
2. Reproduce every valid finding with a failing test before fixing it.
3. Run full pytest, Ruff, strict mypy, retained motor and learning gates, FlyGym parity when
   available, artifact audits, and diff checks.
4. Map each specification claim to code, tests, evidence, and retained outcomes and state all
   remaining experimental-validation limits explicitly.
