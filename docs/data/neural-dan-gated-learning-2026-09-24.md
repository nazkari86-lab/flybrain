# Neural-DAN-gated learning addendum — 2026-09-24

## Change

The retained biological launchers now set
`reinforcement_source=contact_gated_neural_dan`. Physical contact is still an
anonymous sensory gate, but KC→MBON plasticity receives only DAN IDs that were
observed spiking in the same neural window and that belong to the declared
contact valence. A contact can therefore no longer update weights when the
corresponding DAN population is silent.

The legacy `contact_recruited` mode remains available for calibration fixtures,
but the behavioral claim gate rejects it. No planner, reward scalar, target
coordinate, gait policy, or graph mutation was added.

## Retained observations

Using `artifacts/male-cns-v1.0-w5`, seed 7, and the reference body:

| Run | Routed DAN events | Actual DAN spikes | Minimum KC→MBON multiplier | Motor spikes |
| --- | ---: | ---: | ---: | ---: |
| 2 body steps | 0 | 20 | 1.0 | 35 |
| 10 body steps | 8 | 397 | 0.1999983 | 210 |

The two-step case is an intentional negative control: observed DAN activity
alone does not count as appetitive learning when no declared appetitive DAN
spike coincides with contact. Longer windows expose the measured valence-specific
route and update the local overlay.

The retained benchmark remains fail-closed: the fresh single-seed ten-step
behavior artifact still reports `behavioral_claim_allowed=false`, and food and
threat comparisons are not a multi-seed positive result. This change improves
causal provenance; it does not establish autonomous intelligence.

Fresh benchmark artifact: `artifacts/autonomous-behavior-neural-dan-seed7-train1-holdout1-steps10-v2.json`.
It records 17 routed DAN events during normal training, zero routed events in
the four frozen holdout episodes, exact replay, unchanged graph, frozen weights
and memory, and `generalization_verified=true`. The one-seed food deltas against
no-plasticity, DAN-lesion, KC→MBON-lesion and rewired controls were respectively
`-0.0000348`, `-0.0000348`, `-0.0000654`, and `-0.0000810`; these are not positive
learned food-seeking results.

## Verification

- Full local suite after the implementation: 429 passed, 17 skipped.
- Real retained episode and behavior CLI tests after enabling neural-DAN gating:
  2 passed.
- New regression: contact without a real matching DAN spike leaves weights at
  unity.
- New gate regression: a legacy contact-recruited run cannot pass the behavioral
  claim gate even if comparison values are supplied as positive.
- After the latest decoder-readout change: 438 passed, 17 skipped; Ruff and
  mypy passed. The retained MBON assay was also rerun against the saved graph.

## FlyGym physical validation update

The retained FlyGym adapter now uses the complete `LEGS_ONLY` fly geometry,
six distinct foot positions, 0.1 ms MuJoCo substeps within each 10 ms neural
window, and the declared world origin. The benchmark's mass and friction
perturbations now modify the MuJoCo model. The physical report uses measured
actuator force and integrated work after motor limits.

An initial 20-step run with the reference-body decoder authority saturated
618 of 650 nonzero joint commands (95.1%). It delivered only 1.6% of the
requested absolute torque, so its stable body trace was not a useful test of
neural motor amplitude. The FlyGym launcher now declares a lower authority
of `(1e-5, 1e-6, 1e-6)` N·m for the three joint types. This is an engineering
calibration for the FlyGym physics scale, not a measured biological muscle
parameter. The official demo's 65-unit actuator range is retained.

The millimetre-scale arena now places food 0.001004 m from the initial body
centre and scales visual discs, olfaction, and antennae with the same world.
After the DNa02/DNg13 motor-gain change, the 20-step seed-7 run in
`artifacts/autonomous-hexapod-neural-dan-flygym-mm-after-dn-seed7-20.json`
recorded 464 motor spikes across 23 of 36 groups, exact replay, an unchanged
166,606-neuron graph, five final foot contacts, a positive 0.000917 m support
margin, and `fallen=false`. Food distance decreased from 0.001004 m to
0.000934 m, but there were zero food and threat contacts and zero routed
DAN-spike events. This is a stable short physical run, not learned seeking.

The behavioral benchmark now records final physical support for each episode
and requires every normal holdout to finish supported and unfallen before a
behavioral claim can pass. The short single-seed run remains insufficient to
prove autonomous intelligence; independent longer training and positive
appetitive and aversive holdout effects against all controls are still missing.

## MBON-to-descending motor readout

`artifacts/mbon-descending-decoded-torque-seed7-500.json` integrates the
absolute torque decoded from actual motor and descending spikes over each
1 ms window, with zero phase-envelope amplitude. On the retained graph,
normal stimulation produced 49 motor spikes and 0.0001810 N·m·s decoded
torque. Silencing the selected descending populations produced 66 motor
spikes and 0.0001868 N·m·s; the matched MBON control produced 57 motor spikes
and 0.0001948 N·m·s. Source silencing reduced both measures to zero. Thus
stimulation reaches descending neurons, but the selected MBON→DN→motor
specificity gate remains false. Decoded moment is a model readout, not
measured muscle force or evidence of a learned gait.
