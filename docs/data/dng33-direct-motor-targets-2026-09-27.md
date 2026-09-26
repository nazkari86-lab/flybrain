# DNg33 lesion selectively reduces activity in its annotated motor targets in the embodied model

This is a prospective, replicated **model-level biological-circuit result** on
the retained MaleCNS graph coupled to FlyGym. It is not an experiment on a
living fly, a direct-synapse-only proof, or evidence of learned navigation.
The later [annotation audit](dng33-abdominal-motor-identity-2026-09-27.md)
identified all eight direct targets as motor cells with A2–A5 or A7 somata
and AbN-labelled exit nerves, not identified leg-muscle motors.

## Why these cells were tested

The preceding [typed-DN test](typed-descending-lesion-2026-09-27.md) failed
to show a robust effect in the existing 36-population motor decoder. A
retained-graph query then identified eight `vnc_motor` cells reached directly
by both DNg33 neurons: 800659, 803732, 810086, 813291, 814430, 814989,
815205, and 815281. They are annotated MNad03, MNad22, or MNad25. None
belongs to the 182-cell decoder registry; the MaleCNS graph has 708
annotated `vnc_motor` cells in total. Sixteen directed DNg33→target edges
have combined signed graph weight +404 under the project's transmitter-sign
assumption. Anatomy alone could not establish target recruitment, so the
eight IDs and a prospective test were [fixed in a committed design](../superpowers/specs/2026-09-27-dng33-direct-motor-targets-design.md)
(`519b4bc`) before running any new seed.

The [implementation plan](../superpowers/plans/2026-09-27-dng33-direct-motor-targets.md)
was committed as `9b1722d`; readout code and tests were published as
`3726c8f9bf5ac2efbe185bb47685a92f2d0462f1` before seeds 16–18.
The new readout counts emitted Shiu spikes from **all** 708 annotated
`vnc_motor` cells, including zero counts. It does not change the connectome,
sensory input, learning rule, decoder, gains, or body commands. The same
typed-DN silence mask is used during exact replay. There is no hidden RL
policy or gait controller; phase-envelope amplitude was zero.

## Predeclared experiment and result

For each of seeds 16, 17, and 18, the same 100-step, 0.01 s/step FlyGym 2.1.0
episode ran with DNg33 intact, both annotated DNg33 cells silenced, and both
annotated DNg48 cells silenced. The latter is a no-effect intervention
reference, **not** an activity-matched DN control: DNg48 was silent in the
previous panel. All conditions used the same graph, registries, arena,
parameters, and 708-cell readout.

The locked primary gate required in every seed: positive intact spike totals
in both the eight direct targets and the other 700 `vnc_motor` cells; zero
DNg33 spikes after its lesion; at least 20% fewer target spikes; a target
reduction at least 10 percentage points larger than the non-target reduction;
and exact replay, unchanged graph, and identical input revisions.

| Seed | DNg33 spikes intact → lesion | Eight direct-target spikes intact → lesion | Target reduction | Other 700 motor-cell spikes intact → lesion | Other-cell reduction | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 595 → 0 | 550 → 189 | 65.64% | 15,204 → 14,014 | 7.83% | +57.81 pp |
| 17 | 601 → 0 | 551 → 166 | 69.87% | 14,962 → 13,951 | 6.76% | +63.12 pp |
| 18 | 597 → 0 | 540 → 190 | 64.81% | 14,644 → 13,678 | 6.60% | +58.22 pp |

The **predeclared primary gate passed 3/3 seeds**. The DNg48 lesion left the
complete 708-cell spike-count dictionary and per-step motor/body trace
identical to the intact condition in every seed. All nine runs performed
exact replay and left the canonical graph unchanged. The target effect is
not uniform across all eight cells: cell 800659 had only 0/1/1 intact spikes
across the seeds, while the other seven were substantially active. The table
tests the prospectively fixed **sum**, not a claim that every target changed.

This is a causal effect of silencing the annotated DNg33 pair on activity of
their anatomically direct motor targets **in the closed-loop simulation**.
Because lesion and intact bodies follow different trajectories, proprioceptive
and visual inputs can diverge. Polysynaptic paths and body feedback remain
possible mediators; the experiment does not isolate the 16 direct edges.
It also does not validate motor-neuron-to-muscle mapping or real-animal
physiology. Three simulator seeds on one connectome release do not provide a
population-level confidence interval.

## Body and learning diagnostics

The legacy 182-cell registry's motor-spike totals were 2,730 → 2,679,
2,760 → 2,731, and 2,789 → 2,687 for intact → DNg33 lesion. Applied
joint-moment integrals changed from 23.147 → 24.020, 23.842 → 24.673, and
25.311 → 21.387 µN·m·s. Thus the large target-cell effect did **not**
translate into a consistent moment or useful behavioral improvement through
the present partial decoder. Every condition had zero food/threat contacts,
zero contact-gated DAN events, unity KC→MBON multipliers, and at least one
transient `fallen` frame. Learned food seeking, threat avoidance, and stable
biological gait remain unproven.

## Reproduction and immutable artifacts

All nine artifacts carry clean source revision `3726c8f9bf5ac2efbe185bb47685a92f2d0462f1`,
snapshot SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
learning registry SHA-256
`7953875f6df1c655876090b3cd32ca71601e3463f8e5d7789e9f1c47e30c7750`,
motor registry SHA-256
`bd51ac1a9af8f582c66033a2cc7e15d595b799f22f500983ba7d0198f2b2de01`,
and interface registry SHA-256
`9b9ad410b0b86f1f272f94b8ef3d7e4351b8b3bd2f7e9836a52044a39d258e09`.

Use the [README](../../README.md) to acquire/import the four official MaleCNS
products, then run a pair from source revision `3726c8f`:

```bash
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 16 --backend flygym --motor-trace \
  --spike-readout-superclass vnc_motor \
  --descending-type DNg33 --descending-lesion none \
  --output artifacts/repro-dng33-target-normal.json
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 16 --backend flygym --motor-trace \
  --spike-readout-superclass vnc_motor \
  --descending-type DNg33 --descending-lesion annotated_type \
  --output artifacts/repro-dng33-target-lesion.json
```

The published outputs were compressed losslessly with `gzip -k -n -9`.
The following SHA-256 values are of **uncompressed JSON bytes**; check using
`gzip -dc FILE.json.gz | shasum -a 256`.

| Seed | Condition | JSON.gz | JSON SHA-256 |
| ---: | --- | --- | --- |
| 16 | DNg33 normal | [artifact](../../artifacts/dng33-direct-targets-flygym-seed16-dng33-normal.json.gz) | `6518ebee2c5cfeb5417a0951e6a4f29317a5d9ad48c8c5c082efa6aa0904af9b` |
| 16 | DNg33 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed16-dng33-lesion.json.gz) | `ca50f034b447dd78515f54f9dd8b687526f8cbe766aa2666f857ad71e7d571eb` |
| 16 | DNg48 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed16-dng48-lesion.json.gz) | `362879ecdacdad84e8b6d3b26cc8d8ee3264659303809090c5edf68820f74e31` |
| 17 | DNg33 normal | [artifact](../../artifacts/dng33-direct-targets-flygym-seed17-dng33-normal.json.gz) | `76edb68097626ed451abdc50db8d6b1cb04479d52e4f682c3777f970925c10f7` |
| 17 | DNg33 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed17-dng33-lesion.json.gz) | `b77f89f9b9a506582f7b60a59b46804b72268bc0ef039870ee24b4acc225e98a` |
| 17 | DNg48 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed17-dng48-lesion.json.gz) | `6c11445589a6f98defabebc20572062ec35caa221486ebbd6a02af6904c62d35` |
| 18 | DNg33 normal | [artifact](../../artifacts/dng33-direct-targets-flygym-seed18-dng33-normal.json.gz) | `d1c3f5a5984709979f6c74f9d143b46ded3f2df66dfa74cf826d20008785806d` |
| 18 | DNg33 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed18-dng33-lesion.json.gz) | `54a767963281ba415dbd0ab161db8020b80fad358e9e36df12cc9f3237667d36` |
| 18 | DNg48 lesion | [artifact](../../artifacts/dng33-direct-targets-flygym-seed18-dng48-lesion.json.gz) | `6a3c8b17572b5585589ad473fd7daff1d4e0637f4db8b299c54e3ff6ae151727` |

The next mechanistic localization gate is a fixed-source neural replay with
selective DNg33→target-edge removal versus size/weight-matched unrelated
edge removal, followed by the same target-neuron readout. It must be
predeclared separately; this report does not imply that result.
The subsequent [locked edge assay](dng33-direct-edge-localization-2026-09-27.md)
performed that intervention and passed its own model-level gate.
