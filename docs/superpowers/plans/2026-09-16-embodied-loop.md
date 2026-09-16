# Embodied Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, provenance-aware planar world that closes the sensory–MaleCNS–motor–body feedback loop.

**Architecture:** Keep world physics, sensory encoding, motor decoding, and episode orchestration in separate modules. The orchestrator advances the existing Shiu simulator in bounded chunks and feeds only declared external events and emitted output spikes across the boundaries.

**Tech Stack:** Python 3.12, NumPy, Pydantic, existing sparse EventConnectome/Shiu simulator, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-16-embodied-loop-design.md`

## Global Constraints

- Do not create dense `N x N` matrices.
- Do not add CNN, PPO, LLM, A*, object detector, hidden policy network, or target-directed motor shortcut.
- Keep topology immutable; use only existing plastic overlays for learning.
- Validate IDs, bounds, finite values, deterministic seeds, and bounded episode duration.
- Report provenance, replay identity, action/reward/body traces, runtime, and peak RSS.

---

### Task 1: Deterministic planar world and body

**Files:**
- Create: `src/flybrain/embodied_world.py`
- Test: `tests/test_embodied_world.py`

**Interfaces:**
- `ArenaConfig(width: float, height: float, dt_s: float, wall_restitution: float)`
- `FlyBody(x: float, y: float, heading_rad: float, forward_speed: float, angular_speed: float, energy: float, leg_contacts: tuple[bool, ...])`
- `MotorCommand(forward: float, turn: float)`
- `ArenaWorld(config, body, food, threat, wall_segments)` with `step(command) -> WorldStep`
- `WorldStep(body, food_contact: bool, threat_contact: bool, wall_contact: bool)`

- [x] Write failing tests for bounded integration, collision, contact rewards, and invalid commands.
- [x] Run `pytest tests/test_embodied_world.py -q`; confirm missing module failure.
- [x] Implement finite validation, semi-implicit Euler integration, heading normalization, energy drain, and deterministic wall collision.
- [x] Run focused tests and then `ruff check src/flybrain/embodied_world.py tests/test_embodied_world.py`.
- [x] Commit `feat: add deterministic embodied world`.

### Task 2: Sensory encoders and motor decoder

**Files:**
- Create: `src/flybrain/embodied_interfaces.py`
- Test: `tests/test_embodied_interfaces.py`

**Interfaces:**
- `SensoryMap(visual_ids: tuple[int, ...], odor_ids: tuple[int, ...], touch_ids: tuple[int, ...], proprioception_ids: tuple[int, ...])`
- `ExternalEvent(step: int, neuron_ids: tuple[int, ...], voltages: tuple[float, ...], channel: str)`
- `SensoryEncoder.encode(world_step, step: int) -> tuple[ExternalEvent, ...]`
- `MotorMap(left_ids: tuple[int, ...], right_ids: tuple[int, ...], forward_ids: tuple[int, ...])`
- `MotorDecoder.decode(spikes: tuple[int, ...]) -> MotorCommand`

- [x] Write failing tests proving world observations create deterministic sensory events and motor spikes create commands.
- [x] Run focused tests and confirm the missing interface module failure.
- [x] Implement quantized, bounded encoders and spike-count motor voting with an explicit versioned map.
- [x] Run focused tests, Ruff, and mypy.
- [x] Commit `feat: add embodied sensory motor interfaces`.

### Task 3: Closed-loop episode runner

**Files:**
- Create: `src/flybrain/embodied_episode.py`
- Test: `tests/test_embodied_episode.py`

**Interfaces:**
- `EmbodiedEpisodeConfig(max_steps: int, chunk_steps: int, seed: int, sensory_map: SensoryMap, motor_map: MotorMap)`
- `EmbodiedEpisodeResult(passed: bool, replay_exact: bool, graph_unchanged: bool, steps: int, sensory_events: tuple[ExternalEvent, ...], actions: tuple[MotorCommand, ...], rewards: tuple[float, ...], body_trace: tuple[FlyBody, ...], ...)`
- `run_embodied_episode(graph, config, *, plastic_graph=None, world=None) -> EmbodiedEpisodeResult`

- [x] Write failing fixture tests for closed-loop causality, replay, graph immutability, and motor silencing.
- [x] Run focused tests and confirm missing runner failure.
- [x] Implement bounded chunked Shiu calls, output-spike decoding, world stepping, contact-based dopamine, and trace digests.
- [x] Run focused tests and verify replay is exact.
- [x] Commit `feat: close embodied sensory motor loop`.

### Task 4: CLI, artifact, and full verification

**Files:**
- Modify: `src/flybrain/cli.py`
- Create: `tests/test_embodied_cli.py`
- Create: `docs/data/embodied-loop-fixture.md`
- Modify: `README.md`

- [x] Write failing atomic CLI tests for `experiment embodied-loop SNAPSHOT --output RESULT`.
- [x] Implement staged fsync/no-overwrite publication and fixture configuration.
- [x] Run the fixture CLI and save a machine-readable artifact under `artifacts/`.
- [x] Run full pytest, Ruff, mypy, and diff checks.
- [x] Commit `feat: expose embodied loop experiment`.

### Final review

- [x] Inspect every changed file for hidden policy shortcuts, unbounded loops, nondeterminism, and claims stronger than the evidence.
- [ ] Merge the feature branch into `master` only after the full gate passes.
- [ ] Re-run the merged `master` gate and retain the artifact and documentation.
