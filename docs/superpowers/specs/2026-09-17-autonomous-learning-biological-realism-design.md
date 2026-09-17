# Evidence-first autonomous learning and biological realism design

## Goal

Extend FlyBrain from a causally testable MaleCNS sensorimotor platform into an autonomous
associative-learning system without hiding a conventional policy behind biological labels. The
primary agent must learn appetitive and aversive associations through exact Kenyon cell (KC),
mushroom-body output neuron (MBON), dopaminergic neuron (DAN), descending-neuron (DN), motor, and
proprioceptive populations. It must never receive target coordinates, object labels, a scalar
reward, a desired action, or a direct body command.

The project optimizes two separate axes:

1. biological fidelity: exact retained anatomy, local plasticity, explicit unknowns, realistic
   delays and bounded neuromuscular dynamics;
2. autonomous learning: conditioning-dependent behavior that survives lesions, restoration,
   replay, rewired controls, physical perturbations, and unseen-world holdouts.

A successful simulation remains a model result. It is not evidence of animal-equivalent
intelligence or biological validation without independent experimental comparison.

## Current foundation and measured inventory

The existing branch already provides strict retained-snapshot population resolution, a sparse
MaleCNS simulator, retinal and descending interfaces, causal interventions, a deterministic
six-leg/18-joint reference body, exact motor decoding, and proprioceptive observation.

Direct measurements from the retained MaleCNS v1.0 annotation and edge artifacts establish:

- 4,064 cells with `class=Kenyon_Cell`;
- 97 cells with `class=MBON`;
- 340 cells with `class=DAN`;
- 33,496 KC-to-MBON graph edges representing 402,850 annotated synaptic contacts;
- 4,022 KCs and 91 MBONs participating in those KC-to-MBON edges;
- 5,624 DAN-to-KC edges representing 36,149 contacts;
- 1,408 DAN-to-MBON edges representing 36,583 contacts;
- 148 exact motor neurons in 24 trochanter/tibia antagonist populations;
- 622 exact leg proprioceptors in six nerve- and root-side-specific banks.

KC-to-MBON edges have positive canonical transmission signs in the retained graph. DAN-to-KC and
DAN-to-MBON edges have sign zero, so their adjacency is measured but their effect cannot be treated
as an ordinary excitatory or inhibitory current. This distinction is mandatory throughout the
implementation.

## Approaches considered

### Fixed connectome with local KC-to-MBON plasticity (chosen)

Keep canonical topology and all nonplastic weights immutable. Attach a sparse dynamic multiplier
only to existing KC-to-MBON edges. Route a local third factor from exact DAN activity and measured
DAN adjacency. This preserves causal interpretability and permits strict no-plasticity, DAN-lesion,
MBON-lesion, restoration, replay, and topology controls.

### End-to-end deep reinforcement learning on a connectome graph

Training a graph neural controller can produce strong locomotion, but it makes every optimized
parameter a potential hidden policy and does not demonstrate the native computation of MaleCNS.
It is excluded from the primary agent. A separately labeled learned graph controller may be used
only as a performance comparator with equal observations, actions, seeds, and energy budgets.

### Immediate high-fidelity physics plus learning

Starting with FlyGym/MuJoCo would combine neural mapping, plasticity, locomotion, contact, and
dependency failures. The deterministic reference backend remains the causal calibration
instrument. FlyGym becomes the second backend after the neural-to-motor loop passes and must use
the same torque and observation contracts without retraining for the parity claim.

## Immutable graph and sparse plastic overlay

The canonical `EventConnectome` remains unchanged. A `PlasticWeightOverlay` stores only edge indices
for exact KC-to-MBON edges and one bounded multiplier per selected edge. It may not allocate a dense
neuron-by-neuron matrix, create a new edge, reverse an edge, change a transmitter sign, or modify
the canonical CSR arrays.

Every plastic edge is declared by exact pre- and postsynaptic populations plus its retained edge
identity. Registry resolution stores expected IDs, counts, ID hashes, edge count, contact sum, and
source artifact identities. Snapshot drift, unexpected overlap, missing graph IDs, or changed edge
identity fails before execution.

The overlay begins at multiplier one. Updates are bounded and sign-preserving. Resetting the
overlay must recover an exact zero-history state and reproduce the unlearned trace under the same
seed.

## Local three-factor learning rule

KC and MBON activity creates a sparse eligibility trace only on existing active KC-to-MBON edges.
The trace decays with a serialized time constant. Exact DAN spikes supply a third factor only to
eligible edges reached by the predeclared modulatory routing rule. Without both eligibility and
DAN activity, no weight changes.

The first primary rule is bounded depression plus slow recovery:

`m_next = clip(m + recovery - learning_rate * dopamine * eligibility, m_min, 1)`

where `m` is the sign-preserving edge multiplier. Time constants, learning rate, floor, recovery,
and event windows are `model_assumption` values selected before retained-data runs. They are tested
under bounded perturbation and holdout values and never retuned to erase a real null result.

DAN adjacency with sign zero is used only as a modulatory-routing measurement. It never contributes
ordinary membrane voltage. The flat retained artifact does not establish mushroom-body compartment
semantics, so the initial routing interpretations are reported separately:

1. DAN-to-MBON structural routing, where a DAN modulates eligible KC edges terminating at an MBON
   it contacts;
2. a later compartment-specific registry backed by external anatomical evidence and, if required,
   a richer retained synapse/ROI artifact.

Only literature-registered appetitive or aversive DAN types may receive reinforcement events.
Unregistered DANs remain inactive as reinforcement channels.

## Sensory reinforcement boundary

The agent never receives a scalar reward. Anonymous odors and retinal features enter exact sensory
populations. Food or damaging contact produces a bounded sensory event that can recruit a declared
PAM/PPL pathway. Direct DAN stimulation is retained only as a separately named calibration
protocol and may not be reported as full sensory reinforcement.

No encoder may expose food, threat, reward, target, distance-to-target, desired heading, future
state, or desired action. Environment code may use object identity only to choose which physical
contact sensor is activated; identity cannot reach the neural agent.

## Sensorimotor path

The primary behavioral path is:

`anonymous sensory event -> canonical graph -> KC/MBON plastic overlay -> canonical graph -> DN ->
motor populations -> antagonist decoder -> six-leg body -> proprioceptive banks -> canonical graph`

The 148 registered motor neurons control only the two supported antagonist pairs per leg. The
thorax-coxa phase envelope remains an explicit target-independent model assumption until a later
registry revision provides an unambiguous common motor mapping. Descending spikes never become
planar thrust, yaw, foot trajectories, or desired joint angles.

## Autonomous tasks

### Appetitive conditioning

An initially neutral odor is paired with physical food contact. Testing presents the odor without
food in a randomized environment. Positive learning requires a conditioning-dependent preference
or approach effect that is absent in the no-plasticity and relevant-lesion controls.

### Aversive conditioning

A different neutral odor or anonymous visual pattern is paired with damaging contact. Testing
presents the cue without damage in new layouts. Positive learning requires reduced dangerous
contact or cue avoidance with the same causal controls.

Exploration arises only from intrinsic graph activity, deterministic target-independent phase
drive, declared noise, and current sensory/proprioceptive asymmetry. A hidden planner, scripted
turn, direct corrective torque, CNN, PPO, A*, LLM, or privileged state is prohibited in the primary
agent.

## Protocol families

The benchmark keeps these families independent:

1. motor and gait calibration;
2. DN-to-motor open loop;
3. proprioceptive open loop;
4. full sensorimotor closed loop;
5. direct DAN calibration;
6. sensory-to-DAN reinforcement-path calibration;
7. KC-to-MBON plasticity calibration;
8. appetitive conditioning;
9. aversive conditioning;
10. FlyGym backend parity.

Each applicable family includes normal, mirror, matching lesion, opposite lesion, bilateral
lesion, no-plasticity, restoration, exact replay, bounded parameter perturbation, and holdout
conditions. Direct calibrations remain explicitly separated from full-path claims.

## Controls against false connectome advantage

Behavioral comparisons use identical initialization, stimuli, physics, noise, episode budgets,
and energy limits. Controls include:

- fixed canonical graph with plasticity disabled;
- DAN lesion;
- relevant KC-to-MBON edge lesion;
- relevant MBON lesion;
- restoration to exact baseline;
- degree-preserving rewired graph ensemble;
- random behavior with the same movement and energy budget;
- canonical topology with shuffled admissible plastic multipliers;
- unseen odors, layouts, friction, mass, delays, and damaged-leg conditions.

Weak random controls that match only global graph counts are insufficient. Initialization and
degree sequence must be controlled explicitly.

## Physical realism

The deterministic reference backend remains the lesion and replay oracle. Its serialized
assumptions cover joint axes and limits, inertia, damping, moment arms, contact, friction, muscle
authority, delays, activation filtering, fatigue, phase timing, and energy. Published kinematic and
neuromuscular measurements replace assumptions only through a versioned evidence revision.

The motor layer adds bounded activation delay, force saturation, fatigue, and recovery without
changing neuron identity or population authority. Population size normalizes rate rather than
granting larger groups greater force. Proprioceptive events are delayed, bounded, and normalized by
bank size.

FlyGym/MuJoCo implements the same torque-input and joint/contact-observation interface. A transfer
claim requires matching intervention direction, stable execution, and learned preference direction
without backend-specific retraining. Reference-body success is never reported as FlyGym success.

## Success gates and classifications

Every result is one of `positive`, `null`, `directionally_wrong`, or `underpowered`. Thresholds are
fixed before retained runs.

A positive associative-learning claim requires all of:

- sufficient KC, MBON, DAN, DN, and motor activity on the declared full path;
- a conditioning-dependent preference or avoidance effect above threshold;
- no comparable effect with plasticity disabled;
- degradation under matching DAN, MBON, or plastic-edge lesion;
- restoration and exact same-seed replay;
- unchanged canonical topology and nonplastic weights;
- the same effect direction across unseen cue/layout and physical holdouts;
- superiority to fair rewired and random controls;
- bounded energy, stable gait, finite state, and no privileged input.

If the motor path cannot generate sufficient movement, learning behavior is `underpowered`, even if
the KC-to-MBON overlay changes. If weights change without the causal behavioral gates, the result is
a plasticity calibration only, not autonomous learning.

## Artifact and error handling

The immutable experiment artifact records registry and snapshot hashes, exact plastic edge
manifest, initial and final overlay digests, every assumption, schedules, stimuli, spikes, weight
updates, interventions, body traces or digests, classifications, software revision, runtime, and
peak RSS. Publication is staged and atomic and refuses overwrite or input/output aliasing.

Execution fails closed on unknown populations, identity drift, malformed evidence, nonfinite state,
joint-limit escape, topology mutation, sign reversal, dense allocation, invalid intervention,
partial restoration, schedule drift, or output publication failure. A failed run publishes no
artifact.

## Implementation order

1. Complete the evidence-bound motor/gait benchmark and real DN-to-motor/proprioceptive loop.
2. Publish and audit a retained MaleCNS motor artifact without rewriting null outcomes.
3. Register exact KC, MBON, DAN, and KC-to-MBON plastic-edge populations.
4. Implement the sparse eligibility and plastic-weight overlay with exact reset/replay.
5. Add direct DAN and sensory reinforcement-path calibrations.
6. Implement appetitive and aversive conditioning with lesions and fair controls.
7. Add unseen-world and physical holdouts.
8. Add FlyGym/MuJoCo backend parity without retraining.
9. Run a whole-branch scientific, security, and reproducibility review.

This order prevents an autonomous-learning claim from bypassing an unvalidated motor path and keeps
neural, learning, physical, and deployment failures independently diagnosable.

## Research anchors

- MaleCNS v1.0 retained body annotations and canonical edge artifacts provide the measured cell,
  population, and connectivity inventory.
- *The Impact of Structural Changes on Learning Capacity in the Fly Olfactory Neural Circuit*,
  arXiv:2509.19351, supports KC-to-MBON plasticity as the learning locus while motivating explicit
  structural controls.
- *Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly*,
  arXiv:2602.17997, is a performance-oriented deep-RL alternative and not evidence that native
  MaleCNS dynamics learned the behavior.
- *Topological Sensitivity in Connectome-Constrained Neural Networks*, arXiv:2604.04033, motivates
  shared initialization and degree-preserving rewired controls before claiming a connectome
  advantage.
