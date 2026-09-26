# Runner terminal-obstacle curriculum — 2026-09-26

This is a new result for FlyBrain's **engineered, one-button procedural game
learner**, not for the MaleCNS biological controller. A frozen DQN policy
selected without consulting the final test seeds completed **124/128** new
levels, versus **49/128** for the source checkpoint on the identical levels.
There were 75 paired levels completed only by the new policy and zero completed
only by the source policy. Four new-policy failures remain; this is not perfect
mastery or evidence of general/biological intelligence.

## Artifacts and identity

- [Source checkpoint](../../artifacts/games/runner/terminal-curriculum-v1/base/manifest.json)
  at 80,000 steps; model SHA-256
  `6ece5124a7c8b98dd8035a04108304a71b077e48ca062babb90d3b2ac18dcb3a`;
  replay SHA-256
  `1a65afdfce5d0a5cf8575e2b7fac4c501e4d5ea704a436dc4f79eb71e5dd6609`.
- [Selected checkpoint](../../artifacts/games/runner/terminal-curriculum-v1/best/manifest.json)
  at 180,000 steps; model SHA-256
  `8a2829b1b72129f33781258cbab75839a1d84f5d14f6aede893e3878b6cd5a8f`.
  Its replay SHA-256 is
  `dd427aeb4318bb894e87b6228a389b7c6cd0d78f45b406146411cdf7f3ac6268`.
  Both checkpoints contain their replay buffers and exact seed schedules.
- [Per-level evaluation](../../artifacts/games/runner/terminal-curriculum-v1/evaluation-seeds20260927-20260929.json)
  (JSON SHA-256
  `0c7c3c153e5d7d89d434fa551617bad8b7db9a0c014de20ff2714ba21277ff99`).
  It records all 128 seed IDs, outcomes, model hashes and frozen-evaluation checks.
- The base library was clean at source revision
  `7d339b6a6bb13292d576bd34fdf6f07afaee6805`; the experiment-only
  curriculum subclass is reproduced below. Dependencies were Gymnasium 1.3.0,
  stable-baselines3 2.9.0, NumPy 2.5.3 and PyTorch 2.14.0.

## Training and selection

The source checkpoint had passed 23 obstacles in seven of its original eight
holdout levels but collided with the 24th; the eighth ended with a gap and was
completed. Those original holdouts were inspected during development and are
**not** the final evidence. Merely continuing the original full-level training
to 300,000 steps did not improve its four-level validation score. A separate
short-level pretraining attempt also failed to transfer reliably; **it is not
part of the selected policy**.

The successful continuation loaded the source model and replay buffer, then
trained for ten 10,000-step DQN chunks. It kept the original 32 training seeds
and four validation seeds. On alternating training resets, the same 24-obstacle
level began either at the usual origin or grounded at
`x = last_obstacle.x - 2.8`, safely after the preceding obstacle. Rewards,
physics, observations, actions and final evaluation worlds were unchanged.
The seeded training-world cycle restarted at index zero for this curriculum.

After each chunk, the frozen policy was scored on the same four validation
levels. Selection was lexicographic: first the number of completed levels,
then mean normalized distance. The 180,000-step checkpoint reached **4/4**
validation completions, versus **1/4** for the source. A second run of the
same training trajectory reproduced its exact policy-state SHA-256
`6e340103ed0edc22fba638d0d1e2d28c5df3320c99d39a1f673637edeb4e4110`
and training seed index `467` at the selected step.

The essential curriculum reset, using the existing training environment, is:

```python
from flybrain.games.runner_env import RunnerState
from flybrain.games.runner_learning import _SeedCyclingRunnerEnv

class TerminalCurriculumEnv(_SeedCyclingRunnerEnv):
    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.seed_index % 2 == 0:
            self._state = RunnerState(x=self.obstacles[-1].x - 2.8)
            observation = self._observation()
            info = self._info(collision=False, completed=False)
        return observation, info
```

Load `base` with this environment and its `replay.pkl`, call
`model.learn(total_timesteps=10_000, reset_num_timesteps=False)` ten times,
and select each chunk with the four validation seeds in the source manifest.
The source and selected model hashes above identify the exact checkpoints;
the saved `best/manifest.json` and replay allow normal continuation afterward.
The curriculum subclass is an internal experiment protocol, not a claim that
ordinary `games train --resume` recreates the preceding mixed-start training.

## Independent evaluation

Each 64-level batch used `numpy.default_rng(root).integers(0, 2**31, size=64)`.
The seeds are unique and disjoint from training, validation, the original
eight inspected holdouts, and each other. Root `20260927` was fixed before
curriculum training; root `20260929` was chosen as a replication after the
first result, with the model frozen and unchanged. The JSON above contains
their full seed lists and digests.

| Fresh batch | Source 80k | Curriculum 180k | New-only wins | Source-only wins |
| --- | ---: | ---: | ---: | ---: |
| PCG64(20260927), 64 levels | 24 | 62 | 38 | 0 |
| PCG64(20260929), 64 levels | 25 | 62 | 37 | 0 |
| Combined | **49/128** | **124/128** | **75** | **0** |

The candidate's completion rate on this procedural distribution was 96.875%
versus 38.281% for the source; its mean normalized distances were 0.9830 and
0.9850 in the two batches. Both evaluations left their checkpoints unchanged.
The experiment demonstrates a large within-game improvement from targeted
training exposure, not transfer to commercial Geometry Dash, chess, arbitrary
games, a real fly, or the MaleCNS connectome.

Re-evaluate the published frozen checkpoints from the repository root:

```python
from pathlib import Path
import numpy as np
from flybrain.games.runner_learning import evaluate_runner

root = Path("artifacts/games/runner/terminal-curriculum-v1")
for seed_root in (20260927, 20260929):
    seeds = tuple(map(int, np.random.default_rng(seed_root).integers(0, 2**31, size=64)))
    baseline = evaluate_runner(root / "base", seeds)
    candidate = evaluate_runner(root / "best", seeds)
    assert not baseline.training_mutated and not candidate.training_mutated
    print(seed_root, baseline.completions, candidate.completions)
```

Watch the selected policy locally:

```bash
uv run flybrain games watch --game runner \
  --checkpoint artifacts/games/runner/terminal-curriculum-v1/best
```

The standard `games evaluate` command uses the original eight seeds embedded
in the source schedule. Use the linked 128-level JSON, not that command's
default eight, for this result's independent generalization claim.
