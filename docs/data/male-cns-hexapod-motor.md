# MaleCNS hexapod motor assay

## Scope

This assay resolves 36 exact thorax-coxa, trochanter, and tibia motor populations, six exact proprioceptive banks, and
six exact descending populations from the retained MaleCNS v1.0 annotation table. It keeps four
families separate: direct motor/body calibration, target-independent phase-gait calibration,
descending-to-motor recruitment, and proprioceptive plus stateful body feedback.

The canonical 166,606-neuron graph and its sparse edge weights are never changed. Only direct
motor calibration stimulates motor cells; it is not evidence that sensory or descending pathways
produce walking. The thorax-coxa phase envelope and the body mechanics remain explicit model
assumptions.

## Exact retained run

```bash
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/flybrain \
  experiment hexapod-motor \
  /Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
  --registry data/registry/hexapod-motor-registry-v2.json \
  --steps 180 \
  --seed 7 \
  --output /Users/dulatnurlanuly/Downloads/flybrain/artifacts/hexapod-motor-malecns-v2-seed7-180-phase.json
```

The opt-in real-data regression gate is:

```bash
FLYBRAIN_MALECNS_SNAPSHOT=/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
  PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest \
  tests/test_hexapod_motor_real.py -q
```

## Immutable identities

| Field | Value |
| --- | --- |
| Software revision | `c86bdbe` |
| Dataset | `male-cns-v1.0-essential` |
| Registry SHA-256 | `c95d20619d0d7e63e8189213effff02c1b4b5eb0f048b7a53fe778e12e98ae29` |
| Snapshot content SHA-256 | `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f` |
| Annotation SHA-256 | `ead87b4d37f97968f2845d525309649a1db9c7e8a032b147af32d51318eaa67b` |
| Result SHA-256 | `b1d59f1f1ef5f0776652f7cc1a8f5dba21c1693c231fcfec016fdc33d0dc813b` |
| Result size | 3,296,868 bytes |
| Graph | 166,606 neurons; 6,240,402 directed sparse edges |
| Protocol | 180 neural/body chunks; seed 7 |

The generated JSON is deliberately local and ignored under `artifacts/`; the command, identities,
and checksum above make it reproducible without overwriting prior outputs.

## Measured outcomes

| Family | Classification or gate | Measured result |
| --- | --- | --- |
| Direct motor calibration | 36/36 `positive` | Every exact thorax-coxa, trochanter, and tibia population produced its declared antagonist joint effect; bounds, finite state, nonnegative energy, replay, and graph integrity passed. |
| Phase propulsion calibration | `null` walking claim | The target-independent thorax-coxa phase model preserves support and forward displacement, but does not establish alternating tripod support. This is not reported as walking. |
| DNa02/DNg13/MDN to motor | 6/6 `null` | No declared descending population recruited sufficient exact motor spikes through the retained canonical graph at this protocol. |
| Proprioception to motor | 2 `positive`, 3 `null`, 1 `directionally_wrong` | Left and right middle banks passed; left fore, right fore, and left hind were null; right hind failed the homologous direction criterion. |
| Stateful closed loop | `directionally_wrong` | Normal feedback generated 584 proprioceptive and 5 motor spikes with 0.00379985 rad neural joint motion. Matching proprioceptive and motor lesions reduced motor spikes and neural joint motion to zero. |

The closed-loop mirror condition explicitly swaps all six left/right sensory banks and all 36
left/right motor output groups at the interface boundary. It produced 561 proprioceptive and 6
motor spikes with 0.00826899 rad neural joint motion, so the reflected body did not match normal.
This is retained as an observed canonical-graph asymmetry; it is not a broken duplicate-run
control.

## Resource and integrity audit

The final CLI run completed in 42.25 seconds with 994,985,448 bytes peak resident memory on this
host. Canonical graph digest gates passed for direct, descending, proprioceptive, and closed-loop
families. All four replay gates passed. The closed-loop sparse-storage gate passed. No dense
neuron-by-neuron matrix is created by the assay.

## Scientific boundary

- Exact annotation selection establishes the interface populations, not neuron-to-muscle force,
  moment arms, receptor physiology, or an animal-equivalent biomechanics model.
- Positive direct motor tests calibrate the reference body decoder only; they do not demonstrate
  biological gait or descending control.
- The phase gait, all descending pathways, and the full feedback loop did not satisfy a positive
  walking claim on this retained run.
- The observed left/right asymmetry is a simulation result under fixed Shiu dynamics and declared
  interface transformations; it is not an animal behavioral conclusion.
- There are no target coordinates, object labels, scalar rewards, desired actions, PPO, CNN, LLM,
  planner, or hidden locomotion policy in this primary loop.
- This phase does not demonstrate learned food seeking, threat avoidance, autonomous intelligence,
  consciousness, or biological equivalence.

The next phase is the already specified local KC-to-MBON plastic overlay with DAN modulation. It
must retain this exact motor interface while proving learning through causal conditioning and
holdout assays rather than by adding a task-solving controller.
