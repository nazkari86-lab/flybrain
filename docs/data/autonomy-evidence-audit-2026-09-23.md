# Autonomous-learning evidence audit — 2026-09-23

The requested endpoint remains an autonomous adult fly model with learning,
biological sensorimotor control, persistent memory, transfer and causal evidence.
It has not been established. The food/threat benchmark tests only a subset of it.

## Reproduced and corrected defects

1. Benchmark replay was hard-coded true although every episode passed `replay=False`.
   The episode itself compared its first trace with itself when replay was disabled.
2. Replaying a mutable training episode started from its *updated* weights.
3. Holdout episodes copied weights but continued to learn on the copy, so results
   did not isolate knowledge retained from training.
4. Duplicate seeds could be counted as independent replicates.
5. Coincident odor events replaced already scheduled retinal events.
6. The autonomous default added sinusoidal joint torque even with no neural spikes.
   Descending spikes also modulated this external oscillator. The oscillator is
   now disabled by default; it remains an explicit calibration option, and runs
   using it cannot pass the autonomous behavioral claim gate.
7. KC→MBON lesions also disabled DANs, confounding the intervention after DAN
   silencing was added. They now remove only the specified plastic edges and
   disable their plasticity, while leaving DANs enabled.
8. Distance improvement used the first *post-step* distance as the initial value,
   discarding the first action's contribution. The episode now records the actual
   pre-step distances and the benchmark uses those values.

Regression tests: `tests/test_autonomous_evidence.py`. They exercise real episode
dynamics, real benchmark aggregation, a deliberately non-repeatable backend,
and controlled coincident sensory events. The sensory test also directly measures
DAN spikes before and after silencing. Artificial diagnostic stimulation in that
test is a fixture, not part of the adult retained controller.

New evidence protocol: `measured-replay-frozen-holdout-v2`. Every episode records
condition, replicate seed, episode index, training/holdout status, weight-change
status, plasticity state, replay status, graph status and trace digest. Neural
summaries include actual routed DAN spike counts, distinct from the contact-based
reinforcement events used by the plasticity approximation.

## Corrections to earlier interpretations

- Positive food/threat deltas in `behavioral_metrics.compare_conditions` mean
  **normal performs better than the control**. Thus the seed-7 temporal-plume food
  delta of +0.022945 against KC→MBON lesion did not mean the lesion performed better.
  That result was nevertheless single-seed, assisted by an external phase drive,
  and did not measure replay. It cannot establish autonomous learning.
- DAN-lesion and no-plasticity having identical output is not, by itself, an error.
  If these DANs have no relevant spiking contribution, removing their plasticity
signal can produce the same outcome. Comparing either to normal does not isolate
  the effect of the newly added neuron-silencing mask.
- High CPU utilization establishes that a process is computing, not that it is
  making scientific progress. A tool session ID is not an OS PID. Missing output
  alone does not establish a resource failure; inspect the session's terminal
  exit status and the actual process command before restarting a run.

## Performance measurement

Profiling identified repeated DAN routing over all 33,496 plastic edges. Routing
now reduces once per unique MBON (91 in the retained manifest), then gathers back
to edges. The float32 sum uses the same route order as before. A measurement on
the retained routes with fixed seed-7 trace values gave 0.15360 seconds for the
old routing calculation versus 0.00305 seconds for the new one (50.35x for this
operation), with bitwise-equal outputs. This is not a 50x whole-simulator claim.

## Verification

- Full local suite after the changes: **354 passed, 20 skipped**. Skipped tests
  are not counted as biological or retained-data validation.
- Explicitly enabled real integration test with
  `FLYBRAIN_MALECNS_SNAPSHOT=artifacts/male-cns-v1.0-w5`:
  `tests/test_autonomous_behavior_cli_real.py` **1 passed** (56.18 seconds).
  It checks atomic CLI publication, measured replay, frozen holdout weights,
  unassisted output, distinct controls, and zero observed DAN spikes under lesion.
- Ruff and mypy passed. `git diff --check` passed.
- Source fingerprint at this verification point (SHA-256 of the sorted shell
  `shasum -a 256 src/flybrain/*.py` manifest):
  `a918426ff91687e1128bce099cc005a17eac378ce4d4a7a7534c79942f1f7231`.
  This fingerprint covers Python source, not dependencies, datasets or platform.

## Final retained run

Artifact:
[`autonomous-behavior-evidence-v2-seeds7-11-13-10-optimized.json`](../../artifacts/autonomous-behavior-evidence-v2-seeds7-11-13-10-optimized.json).

Command:

```sh
./.venv/bin/flybrain experiment autonomous-behavior artifacts/male-cns-v1.0-w5 \
  --output artifacts/autonomous-behavior-evidence-v2-seeds7-11-13-10-optimized.json \
  --training-episodes 1 --holdout-episodes 1 --steps 10 --seeds 7,11,13
```

The CLI refuses to overwrite the artifact; use a new output name to reproduce.
There are five conditions, three seeds, and four episodes per condition/seed
(food training, threat training, food holdout, threat holdout). Each episode is
repeated once: **60/60** exact replays and **30/30** frozen holdouts. Graph unchanged,
external phase-drive amplitude zero, behavioral claim **false**. Each episode
contains only 0.1 seconds of model time, so this is a verification smoke run.

| Control | Food delta mean [95% bootstrap interval] | Threat delta mean [95% bootstrap interval] |
| --- | --- | --- |
| No plasticity | 0.000013767 [-0.000015151, 0.000029267] | -0.000041489 [-0.000115611, 0.000011663] |
| DAN lesion | 0.000013767 [-0.000015151, 0.000029267] | -0.000041489 [-0.000115611, 0.000011663] |
| KC→MBON lesion | 0.000040037 [0.000010355, 0.000089458] | -0.000012595 [-0.000034474, 0.000030791] |
| Rewired | 0.000017404 [0.000005093, 0.000030589] | -0.000061272 [-0.000072510, -0.000053990] |

Positive deltas favor normal. No evaluated condition reached food or threat during
these short holdouts. Several intervals include zero; the threat comparison with
rewired control is negative throughout. These results do not demonstrate learned
food seeking and threat avoidance.

Routed DAN spike counts in normal holdouts were [243, 211, 243, 207, 240, 203];
under DAN lesion they were [0, 0, 0, 0, 0, 0]. Under no-plasticity they were
[225, 184, 225, 185, 225, 172], despite identical behavioral outputs to DAN lesion.
This directly verifies silencing without equating it to a positive behavioral effect.

Training summaries, holdout neural summaries and final food/threat distances match
the pre-optimization v2 artifact exactly. Whole-run time was 414.22 seconds versus
392.33 seconds for that earlier run. The runs shared local resources differently;
there is **no demonstrated end-to-end speedup** from this wall-time comparison.

## Literature checked online

Metadata and abstracts retrieved from Europe PMC on 2026-09-23 over verified TLS.
These findings motivate model boundaries, not fitted gains or invented synapses.

- Bates et al., *Distributed control circuits across a brain-and-cord connectome*,
  [Nature](https://doi.org/10.1038/s41586-026-10735-w): adult local sensory/effector
  feedback and distributed ascending/descending circuits supervised by learning
  and navigation regions. The abstract does not validate this simulator's dynamics.
- Cheong et al., *Organization of circuits linking descending input to motor output
  in the Drosophila Male Adult Nerve Cord connectome*,
  [eLife](https://doi.org/10.7554/eLife.96084): direct DN→MN connections are infrequent;
  VNC communities support multiple motor systems. The anatomical evidence does
  not justify a direct DN-controlled scripted gait as reconstructed VNC control.
- Jones et al., *Descending neurons integrate learnt information from mushroom
  body with context to promote escape behaviour*,
  [bioRxiv preprint](https://doi.org/10.1101/2025.09.30.679458): larval MBON/context
  convergence through Ipsigoro to Goro. This is a **larval** circuit and a preprint;
  it cannot identify the corresponding adult MaleCNS pathway without further data.

## Remaining requirements of the original request

| Requirement | Evidence still needed |
| --- | --- |
| Full adult CNS fidelity | Coverage audit beyond retained 166,606-neuron/6,240,402-edge subset; validated transmitter/receptor effects and delays |
| Cell-specific biophysics | Measured or justified heterogeneous parameters and independent dynamics validation |
| Multiple senses | Physiological sensory calibration, including wind, taste, and touch; measured plume dynamics |
| Internal state | Hunger, satiety, arousal and other requested state acting through neural modulation, with causal tests |
| Neural motor control | Stable locomotion driven by VNC/MN output without the external phase drive; direct intervention and restoration tests |
| Learned appetitive/aversive behavior | Adequately long preregistered training, independent seeds/worlds, positive causal effects in both tasks, meaningful task success |
| Biological embodiment | Appropriate independent body/physics validation; synthetic backend tests do not establish realistic fly behavior |
| Persistent/continual learning | Save/restore and inter-task transfer of all necessary fast/slow learning state, retention and forgetting tests |
| Generalization | Unseen layouts, perturbations, threats and sensory conditions; no tuning on the final evaluation set |
| Games and task curriculum | Demonstrated sensory-only performance and transfer on the requested tasks (Pong, Flappy Bird, maze, Doom navigation) |
| Causal biological validation | Targeted silencing, restoration and matched controls for multiple behaviors, compared with published experiments |

Short smoke runs measure execution and verification machinery only. Even a
positive food/threat confidence interval would remain a task-specific simulation
result, not a proof of the entire table or of a complete animal intelligence.

## Generalization protocol v4 addendum

The follow-up protocol `measured-replay-persistent-memory-v4` closes one evidence
loophole in this audit. It computes a SHA-256 identity from each world's geometry,
initial body position and physical perturbation, excluding the human-readable name
and scoring target. The behavioral gate now requires zero train/holdout identity
overlap and at least two distinct holdout worlds for both food and threat.

The retained default uses four holdout worlds: mirrored source geometries, shifted
starts, mass/friction changes and one motor-delay perturbation. A fresh three-seed,
two-step smoke artifact contains 90 primary episodes and 90 replays; all 90 replays
were exact, all 60 holdouts froze weights and complete learning memory, and the graph
was unchanged. Structural generalization passed, but every normal-minus-control food
and threat delta was exactly zero. `behavioral_claim_allowed` therefore remained
false. See `docs/data/autonomy-generalization-v4-2026-09-23.md` for exact identities.
