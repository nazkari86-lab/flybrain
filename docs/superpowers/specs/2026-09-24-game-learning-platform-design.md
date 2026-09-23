# FlyBrain Game Learning Platform Design

**Date:** 2026-09-24
**Status:** Approved architecture; awaiting written-spec review
**Scope:** A practical, persistent learning platform with a procedural one-button runner and chess as the first two game adapters

## 1. Objective

Add a game-learning subsystem to FlyBrain that can train without manual play, save and resume its learning state, evaluate on unseen situations, and let a user watch or play against the trained agent.

The first release targets two deliberately different games:

1. A local procedural one-button runner in the style of Geometry Dash, used for fast reactive reinforcement learning and unseen-level evaluation.
2. Standard chess, used for legal-action masking, planning, policy/value learning, and self-play evaluation.

The subsystem is an engineered game AI. It may coexist with FlyBrain's biological MaleCNS experiments, but its RL, search, teacher engine, and game-specific features must remain visibly separate and must never be presented as connectome-generated intelligence.

## 2. Success Criteria

The first release is successful when all of the following are reproducibly demonstrated:

- One CLI namespace controls training, resuming, evaluating, watching, and human play for both games.
- Interrupted training resumes from an atomic checkpoint without losing completed work.
- Runner training improves over a fixed random-agent baseline and is evaluated on procedural seeds never used for training or curriculum selection.
- Chess always emits legal moves, improves over its untrained checkpoint, and is measured against fixed opponents with deterministic time or node limits.
- Every evaluation records the checkpoint hash, code/config identity, seeds, aggregate metrics, and replay locations.
- A live viewer can show the runner and chess games at adjustable speed and can switch between agent and human control.
- Automated tests cover the environment contracts, checkpoint continuation, legal actions, evaluation isolation, CLI behavior, and short learning smoke tests.

These gates prove that the game agents learn within their declared environments. They do not prove general intelligence, biological realism, mastery of every game, or direct compatibility with commercial game clients.

## 3. Considered Approaches

### 3.1 Shared platform with specialized learners — selected

A common adapter, artifact, evaluation, and viewing layer hosts separate learning algorithms. The runner uses a discrete-action RL policy; chess uses legal-action masking, a policy/value model, search, and self-play. This gives the fastest credible result because infrastructure is shared while each game keeps an algorithm suited to its structure.

### 3.2 One pixel-only generalist

A single vision-to-action agent would look more universal, but it would require far more data and compute, learn chess poorly, and make debugging and evidence attribution difficult. Pixel observations remain an optional later runner adapter, not the first proof.

### 3.3 Direct automation of commercial games

Screen capture plus synthetic keyboard or mouse input is fragile, platform-dependent, and difficult to test reproducibly. It can be added later behind the same adapter boundary after the local learning core passes its gates. The first release does not modify or automate Geometry Dash or online chess services.

## 4. Architecture

The new code lives under `src/flybrain/games/` and does not import the biological connectome pipeline.

### 4.1 Shared contracts

`GameAdapter` defines the environment boundary:

- `reset(seed, mode) -> observation`
- `legal_actions() -> action mask`
- `step(action) -> transition`
- `render() -> frame or structured view`
- `snapshot_state()` and `restore_state()` for replay and diagnostics

`Learner` defines the training boundary:

- `act(observation, legal_actions, training) -> action`
- `observe(transition)`
- `update() -> metrics`
- `save(checkpoint)` and `load(checkpoint)`

`Evaluator` runs immutable evaluation manifests. Training and evaluation seed sets are disjoint by construction. Evaluators never write to model weights or replay buffers.

`ArtifactStore` writes versioned checkpoints, metrics, manifests, and replays atomically. It refuses accidental overwrite unless the command explicitly requests resume from a compatible checkpoint.

### 4.2 Shared orchestration

The trainer owns the loop rather than either game:

1. Resolve the game adapter, learner, seed schedule, and curriculum configuration.
2. Create a new run directory or validate a resume checkpoint.
3. Collect transitions and update the game-specific learner.
4. Periodically write an atomic checkpoint and a metrics event.
5. Evaluate against a frozen manifest without updating the learner.
6. Retain the best checkpoint by the game's declared selection metric while preserving the latest resumable checkpoint.

All randomness is derived from a recorded root seed. Wall-clock timestamps may identify runs but may not affect game dynamics, training samples, or evaluation results.

## 5. Procedural Runner

### 5.1 Environment

The local runner is deterministic for a given seed and configuration. A player advances automatically through a generated sequence of ground segments, gaps, and obstacles. The initial action space is binary: wait or jump.

The Markov observation contains normalized distance to the next hazard, hazard type and dimensions, player height and vertical velocity, horizontal speed, grounded state, and limited look-ahead for the following hazard. The renderer creates the visual scene from the same authoritative state, so the displayed run and training transition cannot disagree.

Level generation uses independently named streams for training, validation, and final holdout seeds. Curriculum stages increase speed, narrow safe timing windows, combine hazards, and add controlled observation noise. Promotion occurs only after a minimum rolling success threshold; evaluation never changes curriculum state.

### 5.2 Learning

The first learner uses a maintained discrete-action RL implementation, selected after a bounded dependency and compatibility check. DQN is the default candidate because the initial action space is small and discrete; PPO is retained as the fallback if short controlled benchmarks show materially better stability.

Rewards prioritize sparse task completion:

- positive reward for forward progress and level completion;
- large terminal penalty for collision;
- small action penalty to prevent constant jumping;
- no privileged future-layout reward unavailable in observations.

Reward shaping is fixed before final evaluation. The random baseline and an untrained-network baseline run on exactly the same final holdout manifest.

### 5.3 Evidence

Reported runner metrics include completion rate, normalized distance, collisions per level, actions per second, return, wall time, training steps, and bootstrap confidence intervals across holdout seeds. The release must show improvement over both fixed baselines on unseen levels; no hard-coded target percentage is claimed before measurement.

## 6. Chess

### 6.1 Environment and representation

The chess adapter uses `python-chess` for rules, move generation, terminal conditions, FEN/PGN serialization, and UCI integration. Legal moves are mapped into a fixed policy action space and masked before sampling or search. Illegal moves are treated as implementation failures, not as trainable game outcomes.

The model receives piece planes, side to move, castling rights, en-passant state, and bounded repetition/history features. It predicts a legal-move policy and a position value. The initial network is intentionally compact enough for local training and checkpoint iteration.

### 6.2 Fast bootstrap and autonomous continuation

To maximize strength per unit time, training has two explicit phases:

1. **Teacher bootstrap:** when a compatible local Stockfish binary is available, bounded engine analysis supplies policy targets and value estimates for generated positions. Teacher provenance, version, limits, and position sources are recorded.
2. **Self-play continuation:** policy/value-guided PUCT search generates games, stores training examples, and updates the network. New checkpoints must beat or tie the current accepted checkpoint under a fixed promotion match before replacing it.

If Stockfish is unavailable, self-play can start from an untrained model, but the CLI clearly warns that useful strength will require substantially more compute. Stockfish remains an optional teacher and benchmark; it is not hidden inside the deployed learned policy.

### 6.3 Evidence

Chess evaluation uses fixed seeds and fixed node or time limits. Metrics include legal-move rate, win/draw/loss, score, search nodes, move latency, and estimated Elo with uncertainty against a ladder of reproducible opponents. Opponents include random legal play, a shallow material-based searcher, frozen prior checkpoints, and configured Stockfish levels when available.

No claim of strong chess play is made from self-play loss alone. Promotion requires match evidence against frozen external baselines.

## 7. Persistence and Artifacts

Each run writes to `artifacts/games/<game>/<run-id>/`:

- `run.json`: immutable configuration, dependency versions, root seed, and code identity;
- `latest/`: the newest complete resumable checkpoint;
- `best/`: the checkpoint selected by the declared validation metric;
- `metrics.jsonl`: append-only training and evaluation events;
- `replays/`: compact deterministic runner traces or chess PGNs;
- `evaluations/`: immutable manifests and result bundles.

A checkpoint contains a schema version, game and algorithm identifiers, model state, optimizer state, replay-buffer metadata, curriculum state, counters, and RNG states. Loading rejects incompatible game IDs, action schemas, model shapes, or newer unsupported schema versions. Checkpoint publication uses a temporary sibling followed by atomic rename.

Large replay buffers may use chunked files referenced by hashes rather than embedding all samples in the checkpoint. Missing or corrupt chunks fail closed with a precise diagnostic.

## 8. CLI and User Experience

The top-level Typer application gains a `games` namespace:

```bash
uv run flybrain games train --game runner --steps 200000 --output artifacts/games/runner/run-001
uv run flybrain games train --game runner --resume artifacts/games/runner/run-001/latest
uv run flybrain games train --game chess --games 500 --output artifacts/games/chess/run-001
uv run flybrain games evaluate --game runner --checkpoint artifacts/games/runner/run-001/best
uv run flybrain games evaluate --game chess --checkpoint artifacts/games/chess/run-001/best
uv run flybrain games watch --game runner --checkpoint artifacts/games/runner/run-001/best
uv run flybrain games play --game chess --checkpoint artifacts/games/chess/run-001/best
```

The viewer is a local window and requires no hosted service or account. Runner view controls include pause, speed, reset, seed change, and human/agent toggle. Chess view supports click-to-move, board orientation, undo in human-analysis mode, new game, agent side, and visible search/score information. Training can run headlessly while periodic replays are viewed separately.

The CLI prints the exact output directory, latest checkpoint, best checkpoint, current evaluation metric, and resume command. `Ctrl-C` triggers a final safe checkpoint after the current atomic update completes.

## 9. Dependencies and Reuse Policy

Before implementation, perform a bounded current check of maintained candidates for:

- Gymnasium-compatible environment contracts;
- Stable-Baselines3 or an equivalent maintained runner learner;
- PyTorch compatibility with Python 3.12 and the current macOS architecture;
- `python-chess` licensing and current API;
- local or installable Stockfish availability and UCI compatibility;
- maintained Geometry-Dash-style environments whose license and observation/action semantics fit this design.

Reuse is accepted only when license, maintenance, Python compatibility, deterministic seeding, checkpoint support, and exact capability fit are verified. A third-party demo is not evidence that it satisfies the end-to-end contract. Custom code fills only the gaps: procedural environment, adapters, artifact schema, evaluation isolation, and unified UX.

Game dependencies belong in a `games` optional dependency group so biological experiments and the existing interactive MuJoCo application remain installable independently.

## 10. Error Handling

- Invalid actions are rejected at adapter boundaries; chess never silently substitutes another move.
- NaN or infinite observations, rewards, losses, or model parameters stop training and preserve the last valid checkpoint.
- Resume validates schema, dependency-critical metadata, game configuration, and hashes before mutation.
- Evaluation refuses overlap with declared training seeds or position manifests.
- Missing optional viewer, Stockfish, or accelerator dependencies produce actionable install guidance; they do not corrupt a run.
- Viewer closure does not terminate a headless trainer unless the trainer was launched by that viewer.
- Disk-space checks run before large replay or checkpoint writes.

## 11. Verification Strategy

Implementation follows test-driven development.

### 11.1 Unit tests

- Deterministic runner generation and dynamics for fixed seeds.
- Correct collision, jump, reward, terminal, and curriculum transitions.
- Chess action encoding round-trips every legal move across representative positions.
- Legal-action masks exclude every illegal move.
- Atomic checkpoint round-trip restores counters, policy outputs, optimizer, curriculum, replay metadata, and RNG state.
- Evaluation seed and position manifests cannot overlap training data.

### 11.2 Integration tests

- Short runner training improves a deterministic micro-scenario over its initial policy.
- Short chess update reduces loss on a fixed tiny batch and produces only legal moves.
- Interrupted and resumed runs match uninterrupted deterministic smoke runs where the underlying learner supports deterministic execution.
- CLI train, resume, evaluate, watch dry-run, and play dry-run commands emit valid artifact references.
- Stockfish integration is skipped with an explicit reason when no compatible binary exists.

### 11.3 Release evidence

- Full lint, type checking, and repository test suite.
- Multi-seed runner evaluation on immutable unseen-level manifests.
- Chess matches against frozen baselines under fixed limits.
- Replay inspection for both games.
- One end-to-end resume test from a copied checkpoint.
- Honest summary separating local learning evidence, teacher assistance, search assistance, and untested external-game compatibility.

## 12. Delivery Sequence

The work is split into independently verifiable milestones while keeping one shared design:

1. Dependency/reuse decision record and shared contracts.
2. Procedural runner, baselines, and deterministic viewer.
3. Persistent runner learner, resume, and unseen-level evaluation.
4. Chess rules adapter, action encoding, viewer, and fixed baselines.
5. Chess policy/value training, optional Stockfish bootstrap, self-play, and promotion matches.
6. Unified CLI, artifact summaries, full regression tests, and recorded demonstrations.
7. Only after these gates pass: optional pixel runner and external desktop-control adapters as separate, explicitly platform-dependent work.

This sequence produces a usable learning demonstration early without weakening the final architecture or overstating what has been proved.
