# Learning-memory continuation — 2026-09-23

## Failure and implementation

The prior benchmark persisted only the effective KC→MBON weights. Each new episode
reset KC eligibility, dopamine traces and all four slow DA/NO induction/effect
arrays. A slow effect already folded into the effective weights could then become
part of the new fast base, losing the distinction between the two mechanisms.

`AutonomousLearningMemory` now records the fast base, effective weights, eligibility,
dopamine traces, slow induction/effect states and NO-competent compartments. A
context hash binds it to the graph, transmitter identities, plastic-edge ordering,
DAN routes, reinforcement populations and learning parameters. Restore checks both
the context and the expected effective weights before continuing learning.

Training carries this immutable state within each condition/seed. Both runs of
each replay start from the same state. Evaluation freezes all learning-state
variables; it does not commit any changes back to training. Each new condition
and independent seed starts fresh. Evidence protocol:
`measured-replay-persistent-memory-v3`.

`save_learning_memory` publishes a checkpoint atomically, with no overwrite;
`load_learning_memory` validates a SHA-256 content digest. Loading the file does
not bypass the episode's anatomical/parameter identity checks. Old weight-only
checkpoints remain weight-only; missing traces are not invented retroactively.

## Biological basis and limits

Online metadata/abstract checked via Europe PMC on 2026-09-23:
Aso et al., *Nitric oxide acts as a cotransmitter in a subset of dopaminergic neurons
to diversify memory dynamics*, [eLife](https://doi.org/10.7554/eLife.49257).
NO develops more slowly than dopamine and acts antagonistically, changing retention
and updating. This supports retaining the model's separate state variables; it does
not establish that the current learning-rule parameters recreate the full animal.

The episode boundary still resets membrane voltages, delayed neural events, body
state and muscle activation. There is no modeled elapsed-time gap between episodes.
This change preserves the existing learning mechanism across training episodes;
it does **not** yet provide continuous whole-agent time evolution or prove transfer
between tasks. Separate tests for long-duration behavior and continual learning
remain required.

An additional retained-data audit found 392 dopamine-labeled neurons and 22,777
outgoing edge records, all with sign 0. Thus ordinary signed LIF voltage propagation
from these cells contributes zero in this model. The plasticity pathway is separate
and currently driven by contact-recruited DAN identities, rather than inferred
dopamine release from the recorded DAN spikes. This explains why a DAN lesion and
no-plasticity can have identical motor outcomes even when their observed DAN spike
counts differ. Sign 0 is not evidence that dopamine has no biological effect;
changing it to an arbitrary excitatory sign would not reconstruct neuromodulation.

## Regression coverage

- Fast traces and slow effects survive an episode boundary and JSON serialization.
- Replayed continuation starts from the original memory and remains exact.
- Frozen evaluation leaves both the weights and full learning state unchanged.
- Incompatible routing or mismatched effective weights is rejected.
- Slow modulation is not applied to an already-modulated fast base twice.
- Conditions and independent seeds do not share learning state.
- File checkpoints refuse overwrite and detect changed payloads.

Tests: `tests/test_learning_memory_continuation.py` and the opt-in retained-snapshot
test `tests/test_learning_memory_real.py`.

The explicit retained-snapshot test passed (34.30 seconds): train, atomically save,
load, resume on a freshly loaded graph, verify increased slow dopamine effect,
exact replay and unchanged anatomy. The real behavioral CLI test also passed
(56.37 seconds), including frozen memory at evaluation.

## Python API

```python
from pathlib import Path
from flybrain.autonomous_hexapod_assay import run_retained_autonomous_hexapod_assay
from flybrain.learning_memory import save_learning_memory, load_learning_memory

snapshot = Path("artifacts/male-cns-v1.0-w5")
first = run_retained_autonomous_hexapod_assay(snapshot, body_steps=2, seed=7)
save_learning_memory(Path("artifacts/learning-checkpoint.json"), first.episode.learning_memory)
restored = load_learning_memory(Path("artifacts/learning-checkpoint.json"))
continued = run_retained_autonomous_hexapod_assay(
    snapshot, body_steps=2, seed=11, learning_memory=restored,
)
```

Use a fresh checkpoint filename: existing files are never overwritten. The short
example verifies continuation mechanics, not learned navigation.

## Retained benchmark result

Artifact:
[`autonomous-behavior-memory-v3-seeds7-11-13-10.json`](../../artifacts/autonomous-behavior-memory-v3-seeds7-11-13-10.json).
Seeds 7, 11, 13; ten body steps per episode (0.1 seconds); runtime 432.55 seconds.
All 60 episode replays matched; all 30 holdouts preserved full learning state.
Every condition/seed began with no prior memory; successive training episodes
used the preceding final memory digest. The immutable graph was preserved.

The behavioral claim remains **false**. Normal reached food in 0/6 holdout
observations (these include the separately designated threat task).
Compared to no-plasticity, food delta was -0.000008058 with 95% bootstrap interval
[-0.000022321, 0.000011147]; threat delta was -0.000055250 with interval
[-0.000122994, -0.000004750]. Positive deltas favor normal. Consequently this short
diagnostic does not show beneficial learning, and the negative threat effect must
not be relabeled as success. Preserving state is an implementation improvement,
not evidence that the existing update rule learns the right behavior.

Final local verification: **360 passed, 21 skipped**, Ruff clean, mypy clean.
Two real-data tests were separately enabled and passed: continuation across file
restore and the behavioral CLI. No long-horizon transfer or full-intelligence
claim follows from these tests.
