# Motor-to-muscle reuse audit — 2026-09-25

The target remains an unassisted, learned MaleCNS → motor-neuron → muscle →
body loop. This audit does **not** establish that target. It resolves the next
reuse decision using the retained snapshot and installed FlyGym 2.1.0, without
altering the controller or running an unvalidated walking policy.

## Local coverage, measured from the retained artifacts

The retained `source-annotations.parquet` contains 708 rows annotated
`superclass=vnc_motor`. The v2 motor registry has 36 populations, comprising
182 distinct body IDs; all 182 occur in `neurons.parquet`. The registry groups
these IDs into three antagonistic joint pairs per leg. It is a selected
interface, **not** a complete inventory of the 708 motor neurons.

A conservative name search (`coxa|cox|Tr |troch|Ti |tib|Fe |fem|tars|Ta `,
case-insensitive) finds 85 other `vnc_motor` rows not in the registry:

| Unregistered annotated type | Neurons |
| --- | ---: |
| Fe reductor MN | 20 |
| Sternotrochanter MN | 14 |
| Tr flexor MN | 13 |
| ltm2-femur MN | 12 |
| Ta depressor MN | 9 |
| ltm1-tibia MN | 9 |
| Ta levator MN | 5 |
| Tr extensor MN | 2 |
| Acc. tr flexor MN | 1 |

This is a string-based *candidate* list. It does not prove that every row is
fully traced, active in the retained graph, or assignable to an independent
FlyGym actuator. Conversely, it can miss leg muscles with other names. It is
not a coverage percentage for all fly muscles.

With `JointPreset.LEGS_ONLY`, installed FlyGym 2.1.0 exposes 66 leg joint DoFs
(11 per leg). The biological bridge in `flygym_backend._dof_names()` selects
18 of those (three per leg: thorax–coxa yaw, coxa–trochanterfemur pitch, and
trochanterfemur–tibia pitch). Therefore `36 motor populations` must not be
reported as `all motors` or `all leg DoFs`. Joint torque is still a project
decoder assumption, not measured muscle force.

The published seed-7, 100-step no-lesion FlyGym trace has only 24/36 active
motor groups (`all_motor_groups_active=false`). Across its 100 recorded
activation windows, all six tibia-flexor groups have mean activation below
0.014, while five of six tibia-extensor groups exceed 0.25 (the right-fore
extensor is 0.010). For example, left-fore tibia flexor/extensor means are
0.013/0.695. These values are decoder activations, not measured muscle forces.
They show a strongly imbalanced input that a more realistic muscle body cannot
be presumed to turn into a coordinated gait.

The first two counts can be reproduced without downloading more data:

```bash
.venv/bin/python - <<'PY'
import json
import pyarrow.parquet as pq

rows = pq.read_table(
    'artifacts/male-cns-v1.0-w5/source-annotations.parquet',
    columns=['bodyId', 'superclass', 'type'],
).to_pylist()
registry = json.load(open('data/registry/hexapod-motor-registry-v2.json'))
groups = [p for p in registry['populations'] if p['role'] == 'motor']
ids = {n for p in groups for n in p['selector']['expected_ids']}
print('vnc_motor', sum(r['superclass'] == 'vnc_motor' for r in rows))
print('groups', len(groups), 'registered_ids', len(ids))
PY
```

The installed package's 66 names were inspected through
`NeuroMechFly.add_joints(Skeleton(..., JointPreset.LEGS_ONLY), neutral_pose)`;
the project's 18 names are returned by `flybrain.flygym_backend._dof_names()`.

## Existing components and exact fit

| Source | Reusable now | Boundary |
| --- | --- | --- |
| [MaleCNS v1.0](https://www.janelia.org/project-team/flyem/male-cns-connectome), CC-BY | Whole male CNS topology and motor-cell labels already retained | A connectome is not a calibrated neuron-to-muscle transfer function or body dynamics |
| [MANC motor study](https://elifesciences.org/reviewed-preprints/96084) and [FANC motor atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC11348827/) | Muscle target names, serial homology and anatomical constraints | MANC/FANC identifiers need dataset-specific matching; FANC maps the left T1 motor atlas, not a measured six-leg actuator calibration |
| [FlyGym 2.1 / NeuroMechFly v2](https://github.com/NeLy-EPFL/flygym), Apache-2.0 | Existing full-body geometry, physics and sensors; already installed and used | Default body exposes joint actuators, not an experimentally validated MaleCNS-to-muscle bridge |
| [FlyGym's experimental FlyMimic integration](https://neuromechfly.org/tutorials/6_muscle_imitation), Apache-2.0; [FlyMimic](https://github.com/gizemozd/FlyMimic), Apache-2.0 | 15 Hill-type muscle actuators and spatial tendons for a mechanistic *left-front-leg* assay | Thorax is tethered; only LF is muscle-driven; other legs are passive/locked; no six-leg walking or food seeking follows from it |
| [hello-fly](https://github.com/polatbulut/hello-fly), MIT | Useful crossed-sensory-wiring causal-control design and FlyGym bridge diagnostics | Its FlyWire v783 is female brain only; the author explicitly states that the VNC, CPG, gait and DN-to-joint bridge are hand-written, with ~55× odour contrast amplification |
| [FlyGM](https://arxiv.org/html/2602.17997v3) | External locomotion-performance comparator | It uses a female whole-brain graph, learned graph states/decoder and deep RL. Its successful gait is not native MaleCNS spiking/plasticity evidence and cannot replace our no-hidden-RL pathway |

The [FlyGym tutorial](https://neuromechfly.org/tutorials/6_muscle_imitation)
and [FlyMimic paper](https://arxiv.org/html/2509.06426) explicitly limit
the muscle model to the LF leg. The paper further notes missing body–body and
body–environment contact forces for untethered movement. The built-in model's
15 actuator names were also inspected in its installed MJCF: it includes the
`LFC_sternal_anterior_rotator`, `LFC_sternal_posterior_rotator`,
`LFTibia_flex_93434` and `LFTibia_extensor_93932` muscles, among others. These
names suggest a *candidate* anatomical join to retained MaleCNS labels; they
do not validate the strength or timing of that join.

## Bounded muscle assay completed

`flybrain.muscle_probe.muscle_action()` now makes the four name-matched
left-fore mappings explicit. The other 11 FlyMimic muscles receive their
model's minimum control value, 0.0001. Neither a CPG, a learned policy nor a
body-coordinate command enters the muscle assay. The mapping remains a
cross-sex, population-to-muscle **model assumption**, not a proven individual
motor-neuron innervation or calibrated force law.

The [machine-readable result](../../artifacts/malecns-to-flymimic-lf-open-loop-seed7-steps100-v1.json)
has SHA-256 `e335d5adbdb80b5f83c6805aa57cb436ac07efb0eba5af6d80d617d9b06a02f9`.
It replays the published seed-7, 100-step [no-lesion](../../artifacts/autonomous-hexapod-flygym-lesion-93e2bd8-seed7-steps100-baseline.json.gz)
and [all-motor-lesioned](../../artifacts/autonomous-hexapod-flygym-lesion-93e2bd8-seed7-steps100-all-motor.json.gz)
MaleCNS activation traces, using the same initially posed, tethered LF body.
The retained input JSON SHA-256 values are `1861d6d2ac1207ef58fc77a7345e6d85bb92ae7eb554baf2278779fc793657a5`
and `ff3f9eb6318f7feff03cadc88c913f594bf4a885f2c2e74c69bf941c68021584`.
The control trace has zero spikes and zero activation in all 36 populations.
Each 1-ms MaleCNS window drives 10 muscle-model physics steps at 0.1 ms.

| Observed output | Unlesioned trace | All-motor-lesioned trace | Difference |
| --- | ---: | ---: | ---: |
| LF tibia pitch, final | 0.621234 rad | 1.578510 rad | −0.957276 rad |
| LF coxa yaw, final | −0.179586 rad | −0.206793 rad | +0.027207 rad |
| LF coxa pitch, final | 0.464571 rad | 0.373394 rad | +0.091178 rad |
| Tibia-extensor mean absolute model actuator force | 101.770900 | 5.803949 | — |
| Sternal-posterior-rotator mean absolute model actuator force | 14.142844 | 0.011840 | — |

Both full muscle replays matched the saved JSON results exactly when repeated
on this host. Model force values are simulator outputs, **not** calibrated
newtons or experimentally validated muscle recordings. Passive settling and
minimum muscle activation persist in the control, so a nonzero control force
is expected. This component test establishes that recorded neural activity
can causally alter a *tethered single-leg muscle model* under an explicit
assumed map. It does not close proprioceptive feedback to MaleCNS, move a
free body, prove coordinated gait, or test learning.

Reproduce the two physical replays and compare with the saved artifact:

```bash
FLYBRAIN_MUSCLE_ASSET_TESTS=1 .venv/bin/pytest tests/test_muscle_probe.py -q
.venv/bin/python - <<'PY'
import gzip, json
from pathlib import Path
from flybrain.muscle_probe import replay_muscle_trace

saved = json.loads(Path('artifacts/malecns-to-flymimic-lf-open-loop-seed7-steps100-v1.json').read_text())
for label, path in saved['source_artifact_paths'].items():
    with gzip.open(path) as handle:
        source = json.load(handle)
    trace = [step['activations'] for step in source['episode']['motor_trace']]
    assert json.loads(json.dumps(replay_muscle_trace(trace))) == saved[label]
print('2/2 matched muscle replays')
PY
```

## Decision and next causal gate

Reuse the existing FlyGym/FlyMimic musculoskeletal model first for a bounded
LF-leg experiment. A motor-population-to-muscle mapping must be explicit,
versioned, provenance-tagged and fail closed on unmatched names. A stimulation
test must show (1) a registry-resolved MaleCNS motor population changes its
corresponding Hill-type muscle activation/force and LF joint trajectory,
(2) its exact neural lesion removes that change, and (3) no-spike and
crossed-population controls do not manufacture the same effect. Run this with
real MaleCNS dynamics and exact replay. Do not extrapolate the result to six
legs or to autonomous food/threat learning. The full-body path still needs
validated muscle geometry/innervation for the other legs and independent
behavioral holdouts against no-plasticity, DAN/KC→MBON lesions, rewiring and
passive controls.

No reviewed primary source supplies a ready-made, validated whole-MaleCNS,
six-leg muscle-driven autonomous learner satisfying those gates. This is a
bounded search conclusion, not a proof that none exists anywhere.
