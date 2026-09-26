# Reference-body reachability probe — 2026-09-26

This is an exploratory diagnostic of the retained MaleCNS/Shiu closed loop,
**not** evidence of learned food seeking, threat avoidance, biological scale,
or complete autonomous intelligence. It follows an earlier 100-step benchmark
whose food holdouts began about 0.29 m from the body while observed movement
was usually only millimetres to centimetres.

The new `--arena-scale` option uniformly scales the reference world's food,
threat and start coordinates, contact radius, odor length scale, antenna offset,
and visual-disc radius. It does not scale the body, torque, neural parameters,
or time. The default remains 1.0; non-default scaling is rejected for FlyGym.
This is a deliberately altered task, not a correction of the reference body's
unvalidated physical dimensions. The scale 0.04 was selected **before** this
run from the prior displacement diagnostic; it was not tuned on these outputs.

## Reproduction and identity

```bash
uv run flybrain experiment autonomous-behavior artifacts/male-cns-v1.0-w5 \
  --training-episodes 1 --holdout-episodes 1 --steps 100 \
  --seed 7 --backend reference --arena-scale 0.04 \
  --output artifacts/autonomous-behavior-reachable-local.json
```

The [full compressed result](../../artifacts/autonomous-behavior-reachable-0ac0d0f-seed7-train1-holdout1-steps100.json.gz)
was generated from clean revision `0ac0d0fb306658f8e77fa70db0148e53c4cd75f2`.
Its decompressed JSON SHA-256 is
`0ca31f38aa60e4dcb5f7e94032ccb1200a3aa78227de62165a26d65d36e8bdb8`.
The file records snapshot/registry hashes, condition observations, neural
activity, per-episode replay/weight evidence, and runtime (1,550 s on this
host). The 100-step horizon meets the declared minimum. All 30 episodes
replayed exactly, the canonical graph was unchanged, all holdout weights and
memory were frozen, normal holdouts ended supported and unfallen, the four
holdout worlds were distinct from training worlds, and 30,964 of 33,496
declared plastic edges were structurally rewired in the control.

## Measured behavior

In the normal condition, food holdouts began at 0.01166 and 0.01255 m from
food, beyond the scaled 0.002 m contact radius. Neither made food contact;
their final distances were 0.01452 and 0.03430 m. One of two threat holdouts
made contact at step 30 and cleared it 13 steps later. Normal training had
14 appetitive and 14 aversive body-step contacts, with 25 contact-gated DAN
spike events. The holdout therefore tests a plastically altered neural state,
but it has no successful food-seeking observation.

| Normal minus control | Food score | Threat score | Food contacts, normal/control | Threat contacts, normal/control |
| --- | ---: | ---: | ---: | ---: |
| No plasticity | +0.002181 | −0.501197 | 0/0 | 1/0 |
| DAN lesion | +0.002181 | −0.501197 | 0/0 | 1/0 |
| KC→MBON lesion | −0.003770 | −0.004431 | 0/0 | 1/1 |
| Structural rewire | −0.004797 | −0.001755 | 0/0 | 1/1 |

Scores combine contact outcomes and distance change, so a positive food
score with **zero food contacts** is not food-seeking success. The −0.5 threat
effect against no plasticity/DAN lesion includes the normal-only threat
contact. Each comparison has only **one independent seed**; its bootstrap
interval is a point and cannot establish statistical reliability. The
benchmark correctly reports `behavioral_claim_allowed=false`. The scaled
arena made threat contact measurable but did not rescue the full causal
food-and-threat learning claim.

The putative avoidance MBON 519128, which a separate artificial-stimulation
assay showed can recruit annotated descending neurons, fired **zero** times
in all four normal holdouts. The other directly connected putative avoidance
MBON, 524893, fired only 1, 0, 2 and 7 times; registered MDN-left/right
spikes were 1/0, 0/0, 2/2 and 3/2. These are observations of the current
simulation and an assumed valence assignment, not proof that these cells are
the unique biological avoidance pathway. They make sensory recruitment and
downstream motor specificity the immediate causal questions.

Next, keep the honest baseline and test the specific sensory→MBON→descending→
motor route and joint-level locomotor authority. Do not tune the arena scale
on this outcome or insert a hidden policy to manufacture success.
