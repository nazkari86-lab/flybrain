# Autonomous Behavior Control Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible multi-episode behavioral benchmark that measures learned food-seeking and threat-avoidance against explicit biological controls on reference and FlyGym bodies.

**Architecture:** Keep the existing single-episode neural loop as the only controller. Add isolated modules for perturbations, plastic-state checkpoints, control-condition construction, behavioral metrics, and benchmark orchestration. Every condition receives its own cloned mutable overlay and deterministic seed; the immutable MaleCNS graph is never edited.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy sparse CSR, PyArrow, Typer, pytest, Ruff, mypy, FlyGym 2.1.0, MuJoCo 3.9.0.

**Spec:** `docs/superpowers/specs/2026-09-19-autonomous-behavior-control-suite-design.md`

## Global Constraints

- Primary path remains anonymous sensory fields → sparse MaleCNS/Shiu graph → DAN/local KC→MBON overlay → motor decoder → body → proprioception.
- No CNN, PPO/RL policy, LLM, A*, direct thrust/yaw, target coordinates, desired action, or reward scalar may enter the controller.
- Plasticity may modify only the predeclared positive KC→MBON overlay; canonical edges, signs, weights, and sign-zero DAN routing remain immutable.
- Results must distinguish dataset measurement, model assumption, and simulation observation.
- `behavioral_claim_allowed` remains false unless normal beats every required control on both holdouts with finite non-degenerate metrics.
- Existing output paths, registry paths, and snapshot descendants remain protected from overwrite.

## File Map

- Create `src/flybrain/behavioral_perturbations.py`: validated body/environment perturbations and deterministic variants.
- Create `src/flybrain/plastic_state_checkpoint.py`: versioned sparse overlay persistence and identity validation.
- Create `src/flybrain/behavioral_controls.py`: independent control-condition overlay/route construction.
- Create `src/flybrain/behavioral_metrics.py`: episode metrics, paired deltas, bootstrap intervals, and claim gate.
- Create `src/flybrain/autonomous_behavior_benchmark.py`: multi-episode training/holdout orchestration and artifact model.
- Modify `src/flybrain/autonomous_hexapod_episode.py`: accept bounded perturbations and report contact/distance/time traces without exposing targets to the neural path.
- Modify `src/flybrain/autonomous_hexapod_assay.py` and `src/flybrain/cli.py`: retained-snapshot benchmark command and atomic artifact publication.
- Create focused unit/integration/real tests under `tests/test_behavioral_*.py`.
- Update `README.md` and `docs/data/male-cns-autonomous-hexapod.md` with command, protocol, and claim limits.

### Task 1: Add validated perturbations and trace measurements

**Files:**
- Create: `src/flybrain/behavioral_perturbations.py`
- Modify: `src/flybrain/autonomous_hexapod_episode.py`
- Test: `tests/test_behavioral_perturbations.py`
- Test: `tests/test_autonomous_hexapod_episode.py`

**Interfaces:**
- `BodyPerturbation(friction_scale: float = 1.0, mass_scale: float = 1.0, delay_steps: int = 0, damaged_legs: tuple[int, ...] = ())` validates finite positive scales, non-negative delay, and leg indices 0–5.
- `BehaviorVariant(name: str, food_position_m: tuple[float, float], threat_position_m: tuple[float, float], perturbation: BodyPerturbation)` is deterministic and immutable.
- `apply_perturbation(parameters: HexapodParameters, perturbation: BodyPerturbation) -> HexapodParameters` returns a new parameter object.
- `AutonomousHexapodResult` gains `trace_distance_to_food`, `trace_distance_to_threat`, `first_food_contact_step`, `first_threat_contact_step`, and `time_to_clear_threat_steps`.

- [ ] Write failing tests for invalid scales, invalid legs, deterministic variant generation, and trace fields.
- [ ] Run `PYTHONPATH=src .venv/bin/pytest -q tests/test_behavioral_perturbations.py tests/test_autonomous_hexapod_episode.py` and confirm the new imports/fields fail.
- [ ] Implement frozen Pydantic perturbation models and apply mass/friction/delay/damage through the body/backend contract; keep damage as bounded torque suppression, never a controller command.
- [ ] Record body observations after each backend step and calculate environment-only distances/contact times.
- [ ] Run focused tests plus Ruff and mypy.
- [ ] Commit `feat: add autonomous behavior perturbation traces`.

### Task 2: Add sparse plastic-state checkpointing

**Files:**
- Create: `src/flybrain/plastic_state_checkpoint.py`
- Test: `tests/test_plastic_state_checkpoint.py`

**Interfaces:**
- `PlasticStateCheckpoint(protocol: Literal["plastic-state-checkpoint-v1"], snapshot_content_sha256: str, registry_sha256: str, edge_indices: tuple[int, ...], multipliers: tuple[float, ...], digest: str)`.
- `save_plastic_state(path: Path, overlay: PlasticWeightOverlay, snapshot: Path, registry_sha256: str) -> PlasticStateCheckpoint` writes atomically and refuses aliases/occupied paths.
- `load_plastic_state(path: Path, binding: PlasticEdgeBinding, snapshot: Path, registry_sha256: str) -> PlasticWeightOverlay` validates identity, sorted locations, shape, finite values, and digest before returning a new overlay.

- [ ] Write failing round-trip, tamper, snapshot mismatch, registry mismatch, and occupied-path tests.
- [ ] Run focused checkpoint tests and confirm failure.
- [ ] Implement canonical JSON serialization with sorted keys and temporary sibling publication using the existing safe atomic-output pattern.
- [ ] Ensure loaded overlays cannot alias the caller’s mutable arrays.
- [ ] Run focused tests, Ruff, and mypy.
- [ ] Commit `feat: persist verified sparse plastic state`.

### Task 3: Add independent control conditions

**Files:**
- Create: `src/flybrain/behavioral_controls.py`
- Test: `tests/test_behavioral_controls.py`

**Interfaces:**
- `ControlCondition` literal values: `normal`, `no_plasticity`, `dan_lesion`, `kc_mbon_lesion`, `rewired_control`.
- `build_condition(condition: ControlCondition, binding: PlasticEdgeBinding, learning: AssociativeCalibrationConfig, seed: int) -> ConditionBinding` returns independently owned overlay and learning routes.
- `ConditionBinding.overlay` is a clone; `ConditionBinding.learning` contains the condition-specific DAN routes and exact edge endpoints.

- [ ] Write failing tests proving normal state is mutable, no-plasticity stays at one, DAN lesion recruits no routes, KC→MBON lesion silences only declared overlay edges, and rewiring is deterministic but distinct.
- [ ] Run focused tests and confirm failure.
- [ ] Implement controls without editing `EventConnectome`; rewiring must use a seeded permutation of declared endpoints only and be labeled analysis-only.
- [ ] Run focused tests, full synthetic autonomous tests, Ruff, and mypy.
- [ ] Commit `feat: add autonomous behavior control conditions`.

### Task 4: Add behavioral metrics and conservative statistics

**Files:**
- Create: `src/flybrain/behavioral_metrics.py`
- Test: `tests/test_behavioral_metrics.py`

**Interfaces:**
- `FoodMetrics(contact_rate: float, mean_time_to_contact_steps: float, mean_distance_improvement: float)`.
- `ThreatMetrics(avoidance_rate: float, mean_contact_rate: float, mean_time_to_clear_steps: float)`.
- `BootstrapInterval(mean: float, low: float, high: float, samples: int)`.
- `compare_conditions(normal: Sequence[EpisodeObservation], control: Sequence[EpisodeObservation], seed: int) -> PairedComparison`.
- `claim_gate(comparisons: Mapping[ControlCondition, PairedComparison]) -> bool` requires finite metrics, non-degenerate samples, positive normal deltas for both food and threat, and all required controls.

- [ ] Write failing tests for exact metric values, empty/non-finite samples, deterministic bootstrap, and claim-gate pass/fail cases.
- [ ] Run focused tests and confirm failure.
- [ ] Implement environment-observation-only metrics and deterministic bootstrap resampling with explicit sample counts.
- [ ] Run focused tests, Ruff, and mypy.
- [ ] Commit `feat: add behavioral metrics and claim gate`.

### Task 5: Orchestrate training, holdout, controls, and checkpoint state

**Files:**
- Create: `src/flybrain/autonomous_behavior_benchmark.py`
- Test: `tests/test_autonomous_behavior_benchmark.py`

**Interfaces:**
- `BehaviorBenchmarkConfig(training_episodes: int, holdout_episodes: int, seeds: tuple[int, ...], variants: tuple[BehaviorVariant, ...], controls: tuple[ControlCondition, ...])`.
- `run_behavior_benchmark(graph, binding, config, motor, proprio, reinforcement, body_parameters, backend_factory) -> BehaviorBenchmarkResult`.
- `BehaviorBenchmarkResult` contains config, per-condition training/holdout observations, paired comparisons, checkpoint digests, backend identity, graph digest, and `behavioral_claim_allowed`.

- [ ] Write failing synthetic tests for state persistence across training episodes, unseen holdout variants, independent controls, exact replay, and graph immutability.
- [ ] Run focused benchmark tests and confirm failure.
- [ ] Implement bounded orchestration: train each condition on its own overlay, evaluate holdout from the trained state, and never reuse mutable state between conditions.
- [ ] Add explicit skip records for unavailable FlyGym and failed perturbation gates.
- [ ] Run synthetic benchmark tests, full existing tests, Ruff, and mypy.
- [ ] Commit `feat: orchestrate autonomous behavior benchmark`.

### Task 6: Add retained MaleCNS CLI artifact and real gates

**Files:**
- Modify: `src/flybrain/autonomous_hexapod_assay.py`
- Modify: `src/flybrain/cli.py`
- Create: `tests/test_autonomous_behavior_cli_real.py`
- Modify: `README.md`
- Modify: `docs/data/male-cns-autonomous-hexapod.md`

**Interfaces:**
- CLI: `flybrain experiment autonomous-behavior SNAPSHOT --training-episodes N --holdout-episodes N --seed S --backend reference|flygym --output OUTPUT`.
- Artifact protocol: `retained-autonomous-behavior-benchmark-v1`.

- [ ] Write failing CLI tests for artifact schema, hashes, condition list, claim gate, atomic publication, overwrite protection, and FlyGym unavailable reporting.
- [ ] Run the new real test without implementation and confirm failure.
- [ ] Implement retained registry resolution, NO DAN resolution, benchmark execution, exact provenance, and atomic publication.
- [ ] Run the retained MaleCNS CLI with a bounded smoke configuration and verify all controls.
- [ ] Update docs with measured values and explicit interpretation limits.
- [ ] Run full pytest, Ruff, mypy, diff-check, and retained MaleCNS tests.
- [ ] Commit `feat: publish autonomous behavior benchmark`.

### Task 7: Final verification and handoff

**Files:**
- Modify: any files required by verification only.

- [ ] Run `PYTHONPATH=src .venv/bin/pytest -q`.
- [ ] Run `PYTHONPATH=src .venv/bin/ruff check src tests`.
- [ ] Run `PYTHONPATH=src .venv/bin/mypy src`.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Run the retained MaleCNS smoke and FlyGym smoke with explicit environment variables.
- [ ] Confirm the final artifact has `behavioral_claim_allowed: false` unless every statistical gate is genuinely met.
- [ ] Commit only if verification changes remain, then report exact evidence and remaining scientific limits.
