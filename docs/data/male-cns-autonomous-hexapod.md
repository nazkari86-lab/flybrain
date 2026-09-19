# Retained MaleCNS autonomous hexapod contact loop

## Scope

This assay removes the predeclared reinforcement schedule from the primary path. At every body
step, an anonymous olfactory field stimulates retained sensory neurons, the immutable MaleCNS graph
propagates activity, exact KC-to-MBON edges use a sparse local plastic overlay, registered motor
populations drive the six-leg backend, and body state returns through six exact proprioceptive
banks. Food or threat identity exists only inside the collision boundary that recruits declared
PAM/PPL DANs; it is never emitted as a reward, target coordinate, desired action, or body command.
At any physical contact, a separate source-event stream reaches only the 2,558 bodies resolved as
`superclass=vnc_sensory`, `class=mechanosensory_tactile`. The retained annotation has no
body-surface-to-neuron receptive-field map, so the current equal unlabelled assignment is explicitly
serialized as `uniform_registered_vnc_tactile_assumption`; it does not expose an object label or a
movement instruction.
Measured PPL101/PAM01 nitric-oxide competence adds distinct Aso et al. 2019 fast dopamine and slow
NO effect traces only to KC-to-MBON edges reached by those registered DAN routes. These traces stay
in a mutable overlay; canonical graph weights and sign-zero DAN routing are never rewritten.

## Reproduction

Install the optional high-fidelity backend when FlyGym parity is required:

```bash
uv sync --extra dev --extra physics
```

Run the deterministic reference backend:

```bash
uv run --extra physics flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 \
  --steps 2 \
  --seed 7 \
  --backend reference \
  --output artifacts/autonomous-hexapod-reference-seed7.json
```

Run the same neural controller and torque contract on FlyGym 2.1.0 / MuJoCo 3.9.0:

```bash
uv run --extra physics flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 \
  --steps 2 \
  --seed 7 \
  --backend flygym \
  --output artifacts/autonomous-hexapod-flygym-seed7.json
```

Publication is staged and linked atomically. Existing output, registry aliases, and paths inside
the snapshot are rejected.

## Retained observation on 2026-09-19

- graph: 166,606 neurons and 6,240,402 retained edges;
- sparse plastic manifest: 33,496 exact positive KC-to-MBON edges;
- reference backend: 7 motor spikes across 4 of 24 motor groups;
- physical appetitive contacts: 2 of 2 body steps;
- routed DAN events: 2 of 2 body steps;
- resolved NO-competent DANs: 46 PPL101/PAM01 neurons;
- slow-memory scope: 10,952 unique KC-to-MBON edges reached by NO-competent DAN routes;
- maximum two-step dopamine effect: 7.160399069375233e-07;
- maximum two-step nitric-oxide effect: 7.998822338166445e-09;
- proprioceptive events: 12, one event per leg and body step;
- changed KC-to-MBON multipliers: 10,552;
- graph unchanged: true;
- exact replay: true;
- reference runtime: 9.95 seconds on the retained local host with slow memory enabled;
- FlyGym retained smoke: exact replay, unchanged graph, 7 motor spikes, 4 active groups, 2 physical
  contacts, and 2 DAN events;
- 18-joint reference/FlyGym torque intervention: all 18 response directions matched without
  backend-specific retraining.

Identity hashes for the measured reference run:

- snapshot content: `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`;
- learning registry: `7953875f6df1c655876090b3cd32ca71601e3463f8e5d7789e9f1c47e30c7750`;
- motor registry: `f4bacc27b45b524b24cff4498ac14dbddf316e359066460980149e0087c86b71`.

## Interpretation limits

This is a schedule-free, contact-driven autonomous neural-learning loop. It proves that the retained
sensory, plastic, modulatory, motor, body, and proprioceptive components can execute together on two
physics backends. It does not yet prove learned food seeking, threat avoidance, generalization, or
animal-equivalent intelligence. Only 4 of 24 groups were recruited naturally in this short retained
episode, although the separate direct causal calibration passes all 24 groups. Behavioral claims
remain disabled until conditioned preference/avoidance beats no-plasticity, matching-lesion,
rewired, random, and unseen-world controls.

The retained multi-condition benchmark is available as:

```bash
uv run --extra physics flybrain experiment autonomous-behavior \
  artifacts/male-cns-v1.0-w5 --training-episodes 1 --holdout-episodes 1 \
  --steps 2 --seed 7 --backend reference \
  --output artifacts/autonomous-behavior-reference-seed7.json
```

Its minimum smoke run records all five conditions and leaves `behavioral_claim_allowed` false because
one paired holdout observation is intentionally insufficient for a scientific behavioral claim.

An expanded retained run with seeds `7,11,13`, 1 training episode, 4 holdout episodes, and 10 body
steps produced 12 paired observations across the food and threat holdouts. The bipartite registered-
olfactory channel assumption produced small food-distance differences against some controls, but all
threat-avoidance deltas remained `0.0`; the claim gate therefore stayed false. This is an observed
failure of the current motor/aversive pathway, not evidence of successful threat avoidance.
