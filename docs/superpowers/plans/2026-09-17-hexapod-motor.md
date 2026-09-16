# Evidence-bound hexapod motor implementation plan

**Goal:** Replace direct planar thrust/yaw with exact MaleCNS leg motor populations, a deterministic
phase-aware six-leg body, and closed proprioceptive feedback.

**Architecture:** A strict motor registry resolves six side/nerve banks and two measured
antagonistic joints per leg. A motor decoder converts only registry-backed spikes into bounded
antagonist activations. A replaceable deterministic reference backend integrates three joints per
leg, with thorax-coxa explicitly model-driven and trochanter/tibia neurally driven. A causal runner
keeps direct motor calibration, gait calibration, DN-to-motor propagation, proprioceptive input, and
closed-loop execution separate.

**Tech stack:** Python 3.12, Pydantic 2, NumPy, SciPy CSR, PyArrow, Typer, pytest, Hypothesis, Ruff,
mypy; optional future FlyGym/MuJoCo adapter outside the primary gate.

**Spec:** `docs/superpowers/specs/2026-09-17-hexapod-motor-design.md`

## Global constraints

- Canonical topology and weights remain immutable and sparse.
- No target coordinates, reward, food/threat labels, desired heading, hidden policy, CNN, PPO,
  LLM, A*, or arbitrary motor IDs may reach the primary decoder.
- Motor selection uses exact annotation predicates only.
- `Tr flexor MN` + `Acc. tr flexor MN`, `Tr extensor MN`, `Ti flexor MN` +
  `Acc. ti flexor MN`, and `Ti extensor MN` are the only neural joint channels in v1.
- Thorax-coxa motion is an explicit phase-envelope model assumption, not inferred neural output.
- Direct motor calibration is never reported as DN-driven or sensory-driven gait.
- Real full-path outcomes may be positive, null, directionally wrong, or underpowered.
- All feature work uses RED-GREEN-REFACTOR, focused review, full-suite verification, and immutable
  artifact publication.

## File structure

- Modify `src/flybrain/biological_registry.py`: add exact motor role and `somaNeuromere` selector.
- Create `data/registry/hexapod-motor-registry-v1.json`: exact motor and proprioceptive declarations.
- Create `src/flybrain/hexapod_body.py`: body state, kinematics, contact, support, and deterministic
  reference physics.
- Create `src/flybrain/hexapod_motor.py`: motor maps, activation filters, and antagonist decoder.
- Create `src/flybrain/proprioceptive_interface.py`: privileged-state-free body observation and
  exact neural encoding.
- Create `src/flybrain/hexapod_benchmark.py`: protocols, conditions, lesions, classifications, and
  graph-integrity gates.
- Modify `src/flybrain/cli.py`: atomic `experiment hexapod-motor` publication.
- Create `tests/test_hexapod_motor_registry.py`.
- Create `tests/test_hexapod_body.py`.
- Create `tests/test_hexapod_motor.py`.
- Create `tests/test_proprioceptive_interface.py`.
- Create `tests/test_hexapod_benchmark.py`.
- Create `tests/test_hexapod_motor_cli.py`.
- Create `tests/test_hexapod_motor_real.py`.
- Create `docs/data/male-cns-hexapod-motor.md` and update `README.md`.

---

### Task 1: Extend strict registry semantics for motor anatomy

**Files:** modify `src/flybrain/biological_registry.py`, modify
`tests/test_biological_registry.py`, create `tests/test_hexapod_motor_registry.py`.

1. Write failing tests proving `somaNeuromere` is allowed only as an exact string predicate and
   `role="motor"` is accepted while unknown roles/columns still fail.
2. Add `somaNeuromere` to `ALLOWED_SELECTOR_COLUMNS` and `motor` to
   `PopulationDeclaration.role`.
3. Add fixture declarations for fore/middle/hind leg, left/right, trochanter/tibia flexor/extensor,
   and proprioception selected by `rootSide` rather than absent sensory `somaSide`.
4. Prove selectors reject overlap, side drift, unexpected count/IDs, and missing graph neurons.
5. Run focused pytest, Ruff, mypy, and commit `feat: resolve exact hexapod motor populations`.

### Task 2: Generate and review the retained MaleCNS motor registry

**Files:** create `data/registry/hexapod-motor-registry-v1.json`, modify
`tests/test_hexapod_motor_registry.py`.

1. Query the retained `source-annotations.parquet` and generate exact selectors for 24 motor groups:
   six legs × trochanter/tibia × flexor/extensor.
2. Flexor selectors use exact `in_values` containing the base and accessory type; extensor selectors
   use exact type equality. Every selector fixes `superclass=vnc_motor`, exit nerve, and `somaSide`.
3. Generate six proprioceptive banks using `superclass=vnc_sensory`,
   `class=mechanosensory_proprioceptive`, entry nerve, and `rootSide`.
4. Store exact expected IDs for motor groups and expected counts plus ID hashes for larger sensory
   banks. Bind all claims to the retained annotation artifact and typed model assumptions.
5. Add an opt-in retained-snapshot resolution test and a complete expected-name test.
6. Run focused gates and commit `data: register MaleCNS hexapod populations`.

### Task 3: Deterministic three-joint six-leg body

**Files:** create `src/flybrain/hexapod_body.py`, create `tests/test_hexapod_body.py`.

**Interfaces:** `LegName`, `JointName`, `LegState`, `HexapodBody`, `HexapodParameters`,
`HexapodTorque`, `ReferenceHexapod.step`, `ReferenceHexapod.observe`.

1. Write failing construction, mirror, joint-limit, energy, and exact-replay tests.
2. Define immutable states with finite-value and shape validation. Joint order is thorax-coxa,
   trochanter, tibia; all signs and limits are serialized.
3. Implement deterministic semi-implicit integration with bounded torque, inertia, damping, joint
   stops, three-link forward kinematics, ground contact, bounded friction, thorax displacement, and
   energy use.
4. Compute support count, support polygon margin, alternating-tripod phase, and fall classification.
   Never read world targets or neural labels.
5. Add Hypothesis tests for finite states, bounds, mirror involution, zero-torque rest, and
   deterministic replay.
6. Run focused gates and commit `feat: add deterministic six-leg body`.

### Task 4: Evidence-bound motor spike decoder

**Files:** create `src/flybrain/hexapod_motor.py`, create `tests/test_hexapod_motor.py`.

**Interfaces:** `MotorGroup`, `HexapodMotorMap`, `MotorActivationState`, `HexapodMotorDecoder.decode`,
`motor_population_silence_mask`.

1. Write failing tests for normalized rates, opposite antagonist signs, homologous mirroring,
   population overlap, unknown lesion names, and no direct body command fields.
2. Build the map only from resolved registry populations. Require every group nonempty, disjoint,
   positive, and present in the graph.
3. Decode spikes into bounded low-pass activation. Neural torque is extensor activation minus flexor
   activation. Population size affects normalization, not total authority.
4. Add a separately serialized thorax-coxa phase-envelope assumption and six tripod phases. It may
   modulate only the model channel.
5. Implement exact named motor masks. Silencing never injects opposite activation.
6. Run focused gates and commit `feat: decode leg motor populations`.

### Task 5: Privileged-state-free proprioceptive feedback

**Files:** create `src/flybrain/proprioceptive_interface.py`, create
`tests/test_proprioceptive_interface.py`.

**Interfaces:** `ProprioceptiveObservation`, `ProprioceptiveMap`,
`observe_proprioception`, `ProprioceptiveEncoder.encode`.

1. Write failing tests for monotonic angle/velocity/contact/load channels, mirroring, replay, and
   absence of position, target, food, threat, reward, or desired-action fields.
2. Observe only current joint state and foot contact/load. Normalize all channels to `[0, 1]` using
   declared joint limits and calibration scales.
3. Encode each leg only into its resolved proprioceptive bank, normalize total voltage by bank size,
   and record the calibration boundary.
4. Reject overlapping banks, missing IDs, nonfinite observations, invalid steps, and unknown legs.
5. Run focused gates and commit `feat: close proprioceptive feedback`.

### Task 6: Direct motor and gait causal benchmark

**Files:** create `src/flybrain/hexapod_benchmark.py`, create
`tests/test_hexapod_benchmark.py`.

1. Define immutable protocol, condition, comparison, classification, assumption, and result models.
2. Build all schedules before execution and hash them. Run each flexor/extensor group, its mirror,
   matching/opposite/bilateral lesion, restoration, replay, perturbation, and holdout mechanics.
3. Validate opposite joint signs, matching-lesion effect, joint bounds, alternating support,
   stability, displacement, energy, graph immutability, and exact replay.
4. Add a phase-gait protocol whose target-independent oscillator drives only thorax-coxa while
   direct motor groups validate trochanter/tibia coordination.
5. Predeclare thresholds for minimum motor spikes, displacement, lateral drift, support margin,
   lesion fraction, and holdout agreement before the real run.
6. Use known-positive and zero-edge-null synthetic circuits. Never force a real positive outcome.
7. Run focused gates and commit `feat: benchmark causal hexapod gait`.

### Task 7: Sparse DN-to-motor and proprioceptive protocols

**Files:** modify `src/flybrain/hexapod_benchmark.py`, modify
`tests/test_hexapod_benchmark.py`.

1. Add open-loop DNa02-left/right, DNg13-left/right, and MDN bilateral schedules that enter the
   canonical graph; measure only resolved motor output.
2. Add motor lesions, DN lesions, restoration, replay, perturbation, and mirrored conditions.
3. Add proprioceptive open-loop protocols with normal, side lesion, restoration, replay, and
   homologous mirror.
4. Add closed-loop neural chunks preserving Shiu state: motor spikes step the body; body state
   produces proprioceptive events for the next chunk.
5. Classify direct motor, DN-to-motor, proprioceptive, and closed-loop families independently.
6. Prove topology/weights unchanged and no dense allocation.
7. Run focused gates and commit `feat: close the neural hexapod loop`.

### Task 8: Safe CLI and artifact publication

**Files:** modify `src/flybrain/cli.py`, create `tests/test_hexapod_motor_cli.py`.

1. Add `flybrain experiment hexapod-motor SNAPSHOT --registry REGISTRY --steps N --seed S --output`.
2. Resolve registry and validate graph before execution. Reject occupied output, input aliases,
   output inside snapshot, malformed registry, identity drift, and missing population IDs.
3. Stage beside destination, write with `xb`, flush, fsync, hard-link atomically, and emit JSON only
   after publication.
4. Test execution failure atomicity and absence of partial files.
5. Run CLI/full mypy gates and commit `feat: publish hexapod motor assays`.

### Task 9: Full MaleCNS execution and evidence report

**Files:** create `tests/test_hexapod_motor_real.py`, create
`docs/data/male-cns-hexapod-motor.md`, modify `README.md`.

1. Add an opt-in test against `FLYBRAIN_MALECNS_SNAPSHOT` asserting exact graph counts, registry
   identity, calibration/replay/integrity gates, finite metrics, and valid classifications.
2. Run the retained 166,606-neuron/6,240,402-edge benchmark without rewriting null outcomes.
3. Audit all IDs, assumptions, schedules, conditions, lesions, restoration pairs, holdouts,
   denominators, graph digests, runtime, RSS, and artifact SHA-256.
4. Record exact commands, Git revision, hashes, measured outcomes, nulls, and limitations. Keep the
   ignored generated JSON local and reproducible.
5. Run full pytest, Ruff, mypy, diff check, and commit `docs: validate MaleCNS hexapod motor layer`.

### Task 10: Whole-branch code and scientific review

1. Review exact selectors, motor-sign evidence, model-assumption leakage, decoder inputs, lesions,
   replay independence, threshold preselection, contact mechanics, dense allocation, and atomicity.
2. Reproduce every valid finding with a failing test before fixing it.
3. Re-run full and real-data gates plus artifact audit.
4. Map every spec section to code, tests, and retained evidence. Leave learned food/threat behavior
   explicitly unclaimed and route it to the next specification.
