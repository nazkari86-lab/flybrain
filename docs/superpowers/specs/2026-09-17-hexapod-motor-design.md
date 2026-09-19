# Evidence-bound hexapod motor and proprioception design

## Goal

Replace the biological-steering assay's direct planar body commands with an evidence-bound
motor-neuron-to-joint interface, a phase-aware six-leg body, and closed proprioceptive feedback.
This is the motor foundation required before learned food seeking, threat avoidance, or richer
autonomous behavior can be evaluated honestly.

The phase must demonstrate that MaleCNS descending activity can recruit declared leg motor
populations and that those populations can drive a stable, causal, replayable hexapod model. It
does not claim measured muscle force, complete biomechanics, animal-equivalent gait, or learned
intelligence.

## Why this stage comes next

The retained MaleCNS snapshot contains 708 `vnc_motor` neurons. The canonical annotations expose
side, thoracic neuromere, exit nerve, and named motor types. ProLN, MesoLN, and MetaLN provide exact
fore-, middle-, and hind-leg banks, while names such as tibia flexor/extensor and trochanter
flexor/extensor provide evidence-bounded antagonistic channels. The snapshot also contains
leg-specific mechanosensory and proprioceptive neurons entering through the same leg nerves.

The previous steering phase still converts descending spikes into planar thrust and yaw. Learning
on top of that shortcut would let a reduced decoder hide incorrect motor biology. This phase removes
that shortcut before connecting mushroom-body plasticity to behavior.

## Approaches considered

### Evidence-bound core plus a FlyGym adapter seam (chosen)

Build a deterministic, inspectable three-joint-per-leg reference body and map exact MaleCNS motor
populations into antagonistic joint channels. Keep the physics interface replaceable and add a
FlyGym adapter only after the mapping and causal gates pass. This permits exact replay, cheap
lesions, and scientific inspection while preserving a path to higher-fidelity MuJoCo physics.

### Immediate FlyGym integration

FlyGym offers stronger rigid-body and contact physics, but it is not installed in the current
environment and does not itself prove an exact MaleCNS neuron-to-muscle mapping. Starting there
would combine dependency, physics, mapping, and neural failures in one opaque gate. It remains the
second backend after the reference interface is validated.

### Planar body with a motor-neuron decoder

This would replace arbitrary IDs but retain direct thrust/yaw and no leg phases. It cannot validate
motor coordination, proprioception, support, or gait lesions and is therefore rejected.

## Evidence boundary

Every motor assertion uses the existing four evidence kinds:

1. `dataset_measurement`: exact annotation selectors, IDs, counts, sides, nerves, and graph edges;
2. `experimental_result`: a cited biological claim about a named motor class or gait mechanism;
3. `model_assumption`: joint axes, moment arms, force constants, damping, oscillator parameters,
   contact geometry, and any mapping not supplied directly by the retained dataset;
4. `simulation_observation`: trajectories, contacts, stability, spikes, torques, lesions, and
   benchmark scores produced by a versioned run.

Named motor types constrain channel sign, but their names do not establish measured force,
moment arm, or joint-angle tuning. Those quantities remain explicit assumptions. A positive
reference-body result is not automatically a positive FlyGym or animal result.

## Canonical motor and proprioceptive populations

Registry resolution extends exact selectors to `somaNeuromere` while retaining the current
allowlist and fail-closed semantics. No population may use row order, body-ID ranges, graph degree,
arbitrary truncation, or post-hoc selection.

The first motor registry resolves all six leg banks from:

| Leg | Side | Neuromere | Exit or entry nerve |
| --- | --- | --- | --- |
| left/right foreleg | L/R | T1 | ProLN |
| left/right middle leg | L/R | T2 | MesoLN |
| left/right hind leg | L/R | T3 | MetaLN |

The first registry assigns only the two antagonistic channels that are present under all three leg
nerves and have explicit directional names:

- trochanter flexion: `Tr flexor MN` plus `Acc. tr flexor MN`;
- trochanter extension: `Tr extensor MN`;
- tibia flexion: `Ti flexor MN` plus `Acc. ti flexor MN`;
- tibia extension: `Ti extensor MN`.

Every selector also fixes `superclass=vnc_motor`, exit nerve, and soma side. Population-size
normalization handles measured count asymmetry without dropping cells. Thorax-coxa actuation is a
phase-envelope assumption in v1: types such as promotor, remotor/abductor, femur reductor,
sternotrochanter, and tarsus depressor are retained but receive no invented common sign because the
retained annotations do not provide one uniform antagonistic pair across all six legs. They can be
promoted only by a later evidence revision.

The registry stores every exact ID and count. Unclassified `vnc_motor` cells remain present in the
canonical graph but are not assigned invented muscle semantics. Empty, overlapping, laterality-
mismatched, or snapshot-drifted groups fail before simulation.

Leg-specific proprioceptive banks use `superclass=vnc_sensory`,
`class=mechanosensory_proprioceptive`, the corresponding entry nerve, and side where the canonical
annotation supports it. Tactile banks remain separate. A population that cannot be separated
exactly is reported as unresolved rather than split heuristically.

## Hexapod state and physics boundary

`HexapodBody` contains a thorax pose and six immutable `LegState` records. Each leg exposes three
bounded joint angles and angular velocities, three filtered muscle activations, foot position,
ground contact, load, and gait phase. Joint order and sign conventions are fixed and serialized.

The reference backend uses:

- deterministic semi-implicit integration;
- explicit joint limits, inertia, damping, and bounded antagonistic torque;
- three-link forward kinematics per leg;
- a planar ground with unilateral contact and bounded Coulomb-like friction;
- support-polygon and center-of-pressure diagnostics;
- no target coordinates, reward, object labels, or desired heading in the body or motor decoder.

This backend is a causal instrument, not a complete fly simulator. All dimensions and force
coefficients are versioned assumptions. The backend interface accepts joint torques and returns
joint/body/contact observations so a future FlyGym implementation can replace physics without
changing biological population resolution or benchmark semantics.

## Neural motor interface

`HexapodMotorMap` binds each registry-backed leg and joint direction to one resolved population;
the model-only thorax-coxa phase channel is stored separately and cannot be mislabeled as neural
output. The motor decoder reads only emitted motor-neuron IDs and a declared neural time window.
For each group it computes
population-size-normalized spike rate, applies a bounded activation filter, and forms antagonist
torque as extensor minus flexor activity. No DN spike directly becomes body thrust, yaw, or a foot
trajectory.

The existing DNa02, DNg13, and MDN populations remain neural inputs to the connectome. Their effects
must reach motor neurons through canonical edges in the full-path protocol. Separate direct-motor
calibrations stimulate one declared motor group only to validate signs, lesions, and body mechanics;
they are never reported as descending-path or sensory-path behavior.

## Phase and gait coordination

The reference body includes six bounded phase oscillators. Their initial tripod relationship is a
model assumption:

- tripod A: left foreleg, right middle leg, left hind leg;
- tripod B: right foreleg, left middle leg, right hind leg;
- the two tripods begin half a cycle apart.

Oscillators generate only a neutral stance/swing envelope and cannot read world targets. Motor
spikes scale antagonist activation within that envelope. DNa02/DNg13 activity may modulate
left-right stride amplitude only through predeclared coefficients; MDN may reverse the phase
progression only through a predeclared coefficient. These modulation calibrations remain distinct
from full connectome recruitment.

The primary full-path result does not force a successful gait. If canonical DN activity fails to
recruit sufficient motor spikes, the outcome is `null` or `underpowered`, and direct commands may
not be substituted.

## Proprioceptive feedback

After each body step, `ProprioceptiveObservation` exposes only bounded joint angle, angular
velocity, foot contact, and load channels grouped by leg. The encoder targets only registry-resolved
proprioceptive banks for that leg. It cannot access food, threat, world-object coordinates, desired
actions, reward, or future state.

Joint angle/velocity/contact encoding is an explicit calibration because the retained connectome
does not provide receptor transfer functions. Encoded events record that boundary. Silencing a
proprioceptive bank removes only its neural events; it does not freeze a joint or inject an opposite
signal.

## Protocols

The benchmark keeps these protocols separate:

1. `motor_group_calibration`: direct activation of each antagonist group, matching lesion,
   restoration, mirror, and replay;
2. `phase_gait_calibration`: oscillator plus direct motor drive, tripod timing, support, and
   joint-limit checks;
3. `dn_to_motor_open_loop`: DNa02, DNg13, or MDN drive enters the canonical graph and motor-neuron
   recruitment is measured without body feedback;
4. `proprioceptive_open_loop`: joint/contact sequences enter only resolved proprioceptive banks and
   downstream motor/DN responses are measured;
5. `hexapod_closed_loop`: neural motor spikes step the body and resulting proprioception feeds the
   next neural chunk;
6. `flygym_adapter_gate`: optional parity protocol using identical torque and observation records
   after the reference backend passes.

Every family includes normal, mirrored, matching lesion, opposite lesion, bilateral lesion,
restoration, identical-seed replay, bounded parameter perturbation, and holdout friction/body
parameters where meaningful.

## Success and claim gates

Infrastructure passes only when:

1. all selected motor and proprioceptive populations resolve exactly and exist in the graph;
2. direct flexor/extensor calibration produces opposite signed joint motion, and mirror swaps only
   homologous legs;
3. matching motor lesions remove the declared torque without creating opposite activity;
4. joint limits, energy bounds, finite-state checks, and contact invariants hold;
5. tripod phase offsets and replay are exact under fixed seed and parameters;
6. proprioceptive observations change monotonically with their calibrated physical variables and
   contain no privileged world state;
7. graph topology and canonical weights remain unchanged;
8. output publication is atomic and provenance-bound;
9. no dense neuron-by-neuron allocation exists.

Behavioral classifications remain `positive`, `null`, `directionally_wrong`, or `underpowered`.
A positive walking claim additionally requires forward displacement, bounded lateral drift,
alternating support, matching motor-lesion degradation, restoration, replay, and the same effect
direction across holdout friction and body parameters. A positive steering or retreat claim
requires the expected mirrored sign and matching DN or motor lesion. Thresholds are fixed before
the retained run.

## Testing and retained artifacts

Fixture tests use tiny exact motor/proprioceptive registries and synthetic circuits with known
positive and null paths. Property tests cover joint bounds, mirror symmetry, energy, deterministic
integration, support, serialization, and absence of privileged fields. Full-data tests resolve all
selected MaleCNS IDs and run sparse DN-to-motor and short closed-loop protocols.

The CLI publishes a single immutable artifact containing registry and snapshot identities, all
assumptions, population manifests, phase/joint/contact traces or digests, interventions, effects,
classifications, graph digests, software revision, runtime, and peak RSS. It refuses overwrite and
publishes nothing after any validation or execution failure.

## Decomposition toward intelligence

This specification deliberately covers motor embodiment only. After its gates pass, the next
separate specification connects olfactory/visual cues, DAN reinforcement, KC-to-MBON plasticity,
and the validated hexapod motor interface in food-seeking and threat-avoidance tasks. That learning
phase must use baseline, no-plasticity, DAN lesion, MBON lesion, restoration, replay, perturbation,
and unseen-world holdouts. No hidden policy, target coordinates, A*, PPO, CNN, or LLM may generate
the primary agent's actions.
