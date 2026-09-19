# Autonomous behavior and scientific control suite

## Goal

Extend the retained MaleCNS hexapod system from a verified short closed-loop episode into a
reproducible multi-episode behavioral benchmark. The suite must test whether local plasticity and
measured modulatory routes produce behavior that generalizes beyond the training episode. It must
not introduce a hidden policy, scalar reward controller, target coordinates, or changes to the
immutable MaleCNS graph.

## Scope and boundaries

The primary path remains:

```text
anonymous sensory fields + body state
    -> Shiu/MaleCNS sparse graph
    -> registered DAN recruitment and local KC->MBON overlay
    -> 24-group motor decoder
    -> reference or FlyGym body
    -> contact and proprioceptive feedback
```

The benchmark may know object locations for environment construction and scoring, but those values
must never enter the neural input, motor decoder, or learning rule. The neural result must expose
only observations, contact events, motor activity, body state, and plastic state summaries.

## Components

### 1. Episode configuration and persistence

Add a versioned configuration for episode count, seeds, arena variants, body perturbations, and
training/holdout splits. Add an atomic plastic-state checkpoint containing snapshot identity,
registry identities, edge locations, multipliers, and a digest. Loading must fail closed on any
identity or shape mismatch.

### 2. Behavioral environments

Provide deterministic food-seeking and threat-avoidance arenas. Training and holdout variants use
different positions and combinations of odor length scale, contact radius, friction, mass, timestep
delay, and damaged-leg masks. The environment generates anonymous odor, touch, taste/contact, and
proprioception signals. It never emits a target or action request to the brain.

### 3. Control conditions

Every benchmark family runs the same seeds and worlds under:

- `normal`: full graph and learning;
- `no_plasticity`: overlay fixed at one;
- `dan_lesion`: declared appetitive/aversive DAN recruitment removed;
- `kc_mbon_lesion`: declared plastic edges silenced while anatomy remains immutable;
- `rewired_control`: deterministic permutation of declared plastic endpoints for analysis only.

Controls must be explicit in the artifact and cannot silently share mutable state with the normal
condition. Rewired controls are not presented as biological predictions.

### 4. Learning and evaluation protocol

Training runs for a bounded number of episodes and persists state between episodes. Holdout runs
start from the resulting state but use unseen worlds and seeds. Food metrics include contact rate,
time-to-first-contact, and distance/contact improvement relative to the initial episode. Threat
metrics include avoidance/contact rate and time-to-clear after threat exposure. Metrics are based on
environment observations, not hidden neural labels.

The suite reports paired deltas against each control, bootstrap confidence intervals across seeds,
and a conservative `behavioral_claim_allowed` flag. The flag remains false unless normal beats all
required controls on both food and threat holdouts with finite, non-degenerate observations.

### 5. Physical robustness and backend parity

Run the same benchmark configuration on the reference backend and FlyGym when available. Record
backend identity, torque traces, body traces, replay status, and graph digest. Include bounded
perturbation families for friction, body mass, neural/body delay, and one-leg damage. No backend-
specific retraining is allowed for the parity comparison.

### 6. CLI and artifact

Add a command that accepts a retained snapshot, registries, seed, episode count, backend, and output
path. Publish one atomic JSON artifact containing protocol version, exact configuration, snapshot and
registry hashes, condition results, per-seed metrics, control comparisons, confidence intervals,
plastic-state digests, backend evidence, and failure/skip reasons. Existing outputs, registry paths,
and snapshot descendants remain protected from overwrite.

## Error handling

Reject invalid perturbation ranges, duplicate IDs, unknown conditions, missing registries, snapshot
identity mismatches, non-finite metrics, and incompatible backend traces. If FlyGym is unavailable,
the reference artifact may complete but must record FlyGym as an explicit unavailable gate rather
than claiming parity.

## Testing strategy

Use test-first development in layers:

1. unit tests for checkpoint integrity, perturbations, controls, metrics, and confidence intervals;
2. synthetic six-leg integration tests proving normal/control separation and exact replay;
3. FlyGym parity tests without retraining;
4. retained MaleCNS tests for full graph identity, all registered routes, and atomic CLI output;
5. fresh Ruff, mypy, full pytest, diff-check, and real retained-snapshot verification before any
   completion claim.

## Explicit non-goals

This suite does not claim animal-equivalent intelligence, consciousness, real-world deployment, or
scientific validation beyond the recorded simulation observations. It does not add CNNs, PPO/RL
policies, LLMs, A*, direct thrust/yaw commands, or a reward scalar to the primary controller.
