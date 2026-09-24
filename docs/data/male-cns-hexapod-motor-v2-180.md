# Retained MaleCNS hexapod motor protocol v2: 180-step causal window

## Why the protocol window changed

The earlier 90-step retained run was underpowered for four descending populations. A
read-only sweep on the unchanged graph showed that the same 10 mV source protocol produced
zero motor spikes for DNg13/MDN at 90 steps but recruited all six registered descending
populations by 180 steps. The change is a declared temporal-window correction, not a change
to the graph, motor decoder, threshold, or biological IDs.

This is consistent with the biological division of labor described in current literature:
leg movement is rhythmic and generated in circuits outside the brain, while descending
neurons modulate steering, limb gestures, speed, and retreat. It is therefore inappropriate
to treat a short single impulse window as a complete DN-to-motor test.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/flybrain experiment hexapod-motor \
  artifacts/male-cns-v1.0-w5 \
  --registry data/registry/hexapod-motor-registry-v2.json \
  --steps 180 --seed 7 \
  --output artifacts/hexapod-motor-malecns-v2-seed7-180-mirror-corrected.json
```

## Retained identities

| Field | Value |
| --- | --- |
| Dataset | `male-cns-v1.0-essential` |
| Graph | 166,606 neurons; 6,240,402 directed sparse edges |
| Motor registry | `hexapod-motor-registry-v2` |
| Result SHA-256 | `3694cf877e76304bdcd147a39e15378aabd8ba9a7721c33c65136e809e49c149` |
| Seed / window | `7 / 180` |
| Protocol | `hexapod-motor-v1` |

## Measured causal families

| Family | Result |
| --- | --- |
| Direct motor calibration | Positive for all 36 canonical antagonist groups |
| DNa02 left/right | Positive; 10 / 6 motor spikes; DN and motor lesion fractions 1.0 |
| DNg13 left/right | Positive; 4 / 5 motor spikes; DN and motor lesion fractions 1.0 |
| MDN left/right | Positive; 5 / 7 motor spikes; DN and motor lesion fractions 1.0 |
| Proprioceptive banks | 4 positive, 2 null: both fore banks remain under-recruited |
| Stateful closed loop | Positive; 11 motor spikes, 1,166 proprioceptive spikes |
| Phase propulsion calibration | Stable positive forward displacement with six supporting legs; walking claim remains null because alternating tripod support is not established |

All causal families pass exact replay and graph-integrity controls. The artifact does not
claim that a few recruited motor spikes constitute walking.

The corrected decoder applies the sagittal mirror sign to thorax–coxa neural torque:
the retained left/right fore anterior calibrations reached `+0.8/-0.8` radians.
Direct calibration still passes for all 36 groups; all six DN paths remain positive,
four proprioceptive banks remain positive, and the closed loop remains positive.
The phase-gait claim remains null because alternating support is not established.
The earlier `...180-phase.json` artifact predates this mirror correction.

## Internet evidence used for model boundaries

1. Yang et al., *Fine-grained descending control of steering in walking Drosophila*, Cell
   (2024), [PMID 39293446](https://pubmed.ncbi.nlm.nih.gov/39293446/) and
   [Europe PMC full text](https://europepmc.org/articles/PMC12778575). The study reports
   distinct descending neurons evoking different limb gestures, including asymmetric stride
   modulation; this supports a temporal, population-level DN assay rather than a direct planar
   command decoder.
2. Sen et al., *Moonwalker Descending Neurons Mediate Visually Evoked Retreat in Drosophila*,
   Current Biology (2017), [PMID 28238656](https://pubmed.ncbi.nlm.nih.gov/28238656/).
   This is the retained MDN retreat evidence; it does not by itself identify every VNC motor
   synapse or prove a complete visual-to-body simulation.
3. Wang-Chen et al., *NeuroMechFly v2: simulating embodied sensorimotor control in adult
   Drosophila*, Nature Methods (2024), [DOI 10.1038/s41592-024-02497-y](https://doi.org/10.1038/s41592-024-02497-y),
   and the [official FlyGym repository](https://github.com/NeLy-EPFL/flygym). These provide
   the current body, retina, odor, contact, and mechanosensory embodiment reference; they are
   not evidence that the retained MaleCNS dynamics learned locomotion.
4. Aso et al., *The neuronal architecture of the mushroom body provides a logic for associative
   learning*, eLife (2014), [DOI 10.7554/eLife.04577](https://elifesciences.org/articles/04577).
   The full-text circuit map supports compartmental KC–MBON plasticity modulated by DANs,
   which remains the learning locus in this project.
5. Jin et al., *Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in
   Fruit Fly*, [arXiv:2602.17997](https://arxiv.org/abs/2602.17997). The abstract explicitly
   describes deep reinforcement learning; it is a useful external performance comparison,
   not proof of autonomous learning in native retained MaleCNS dynamics.

## Remaining gate

The full autonomous-intelligence claim remains disabled. The next required causal bottleneck is
the two fore-leg proprioceptive paths and stable gait under unseen worlds. Learned food seeking
or threat avoidance may only be claimed after multi-seed holdouts beat no-plasticity, DAN,
KC–MBON lesion, rewired, restoration, and random controls without privileged coordinates or a
hidden policy.

## Long-horizon stability and navigation diagnostic (2026-09-23)

The reference body now includes a bounded passive joint-restoring term
`k * (initial_joint_angle - joint_angle)`. This is a physical body parameter, not a neural
command or target coordinate. It is motivated by insect neuromechanical modelling showing that
passive elastic forces and viscous damping are necessary to reproduce loaded leg movements and
that passive stiffness can be comparable to active muscle torque (Mamiya et al., *J Neurosci*
2006, [DOI 10.1523/JNEUROSCI.0161-06.2006](https://doi.org/10.1523/JNEUROSCI.0161-06.2006)).

The 6-second retained diagnostic (`600` body steps) now ends with six supporting legs,
`support_margin_m=0.4041`, `fallen=false`, `motor_spikes=14196`, and exact replay plus an
unchanged graph. This repairs long-horizon mechanical collapse, but it does not establish
learned navigation.

The subsequent 3-seed, 2-holdout, 100-step causal benchmark is preserved at
`artifacts/autonomous-behavior-reference-seeds7-11-13-train1-holdout2-steps100-passive-elasticity.json`.
It has `behavioral_claim_allowed=false`; every food and threat interval crosses zero. The
remaining model limitation is explicit: the current arena emits one scalar odor intensity to
each receptor channel, so it does not yet provide bilateral concentration/flow evidence for
steering. This matters because the arthropod olfactory-navigation review identifies comparison
across sensors and integration over time as core navigation strategies (Steele, Lanz & Nagel,
*J Comp Physiol A* 2023, [DOI 10.1007/s00359-022-01611-9](https://doi.org/10.1007/s00359-022-01611-9)).
The current result therefore proves stable embodied activity, not autonomous intelligence.

## Closed-loop descending modulation

The autonomous episode now passes registered DNa02, DNg13, and MDN spikes into a bounded
phase-envelope modulator before motor decoding. The modulator changes only side-specific phase
bias and phase gain; it has no arena labels, target coordinates, reward, desired action, or
planar command. In decoder v3 the same bounded gain also scales already-present neural motor
activation, so descending spikes have a causal effect even when the external phase-envelope
amplitude is zero; no torque is generated without motor-neuron activation. This remains a model
assumption for the VNC-like interface, while the underlying MaleCNS graph remains immutable.
The one-seed retained regression run after this change measured a different motor/body trace,
but the benchmark still has no full behavioral claim: it is short, single-seed evidence and
must be re-evaluated with independent held-out worlds.
