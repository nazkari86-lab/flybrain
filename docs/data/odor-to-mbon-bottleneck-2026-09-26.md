# Odor-to-MBON recruitment bottleneck — 2026-09-26

This is an exploratory, isolated-neural diagnostic of the retained MaleCNS/Shiu
model. It is **not** a body, learning, or autonomous-intelligence result. The
[compressed result](../../artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz)
was generated from source revision
`aab037153fb70aef90e082e87ec596905600d59d`; the SHA-256 of its
decompressed JSON is
`3f3493acb4b61fa8cedb2f5cc8cb07c84b4367335161e2a9a7045c7c35f3864e`.
The result embeds snapshot and learning-registry hashes. The retained snapshot
and registry are the same inputs identified in the
[20-step closed-loop result](autonomy-reachability-probe-2026-09-26.md).
No graph edges or controller parameters were changed. A subsequent source-order
normalization makes the standalone runner consistent with the retained wrapper
used for this artifact; the artifact's recorded source revision remains exact.

## Stimulus and readouts

The existing `OlfactoryReceptorMap`, `task_odor_assignment`,
`poisson_voltage_events`, `simulate_shiu`, and KC→MBON manifest were reused.
Food comprised both sides of ORN_DM1 and ORN_VA2 (148 source neurons); threat
comprised both sides of ORN_DA2 (41). Each condition ran 2,000 neural steps
at `dt_ms=0.1`, `refractory_ms=2.0`, `synaptic_delay_ms=1.0`, and the default
150 Hz source-equivalent Poisson rate. Each seed (7, 8, 9) had no-source,
food-only, threat-only, and food-plus-threat conditions. A separate
**artificial upper-bound control** set only the 108
declared KC→MBON 519128 edge multipliers to the model's maximum 2.0 during
threat-only stimulation. It is not a learned state. Each condition started
from a fresh Shiu state and used the same seed for its input events and state.
There was no vision, proprioception, body, tactile input, or DAN plasticity.
Seeds 8 and 9 were added after inspecting seed 7, so this is an exploratory
replication check, not a preregistered or statistical test. The packaged probe
checks exact neural replay, paired source events, and unchanged graph/overlay;
all four flags are true in the linked artifact. Earlier one-off figures are
superseded: their receptor-ID tuple ordering changed which neurons received
deterministic Poisson draws.

The fixed snapshot has 166,606 neurons and 6,240,402 directed edges. MBON
519128 has 108 positive declared plastic KC inputs among 245 total incoming
edges; MBON 524893 has 105 among 388. Directed paths from registered
olfactory neurons to both MBONs exist within three hops, but reachability is
not functional recruitment. The declared DAN→MBON manifest gives MBON 519128
18 appetitive and **zero aversive** routes. MBON 524893 has 27 appetitive and
one aversive route. Thus the current valence-compartmental rule has no direct
aversive-DAN route by which threat reinforcement can potentiate the 519128
KC compartment.

| Stimulus | Seed | Source events | Spikes from plastic KCs into 519128 | 519128 spikes | 524893 spikes | Peak 519128 post-step voltage, mV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 7 | 0 | 0 | 0 | 0 | −52.000 |
| Food | 7 | 4,288 | 32 | 0 | 0 | −51.806 |
| Threat | 7 | 1,209 | 28 | 0 | 1 | −51.760 |
| Food + threat | 7 | 5,512 | 32 | 0 | 0 | −51.885 |
| Threat, KC→519128 at 2× | 7 | 1,209 | 28 | 0 | 1 | −51.760 |
| Threat | 8 | 1,190 | 27 | 0 | 1 | −51.331 |
| Threat, KC→519128 at 2× | 8 | 1,190 | 27 | 0 | 1 | −51.331 |
| Threat | 9 | 1,192 | 33 | 0 | 0 | −51.675 |
| Threat, KC→519128 at 2× | 9 | 1,192 | 33 | 0 | 0 | −51.510 |

The spike threshold is −45 mV. `Peak post-step voltage` is sampled after
each Shiu step, including reset on a spike; it is a subthreshold readout for
the non-spiking 519128, not a substitute for counting spikes. For seed-7
threat-only, summing each presynaptic spike count times its retained signed
edge weight into 519128 gave +3,306 positive and −11,195 negative
weighted-spike units. The artificial 2× KC→519128 intervention raised the
positive total to +3,579, with unchanged negative total and zero 519128
spikes. Across seeds 8 and 9, baseline/max2 positive totals were
+3,199/+3,455 and +3,689/+4,004; negative totals were −10,175 and −11,953,
respectively. These are **not** integrated membrane currents, because
synaptic timing, delays, and conductance dynamics matter. They motivate a
time-resolved causal inhibition test rather than prove inhibition alone is
the cause.

The full 20-step closed-loop artifact independently recorded zero 519128
spikes in both normal training episodes and all four normal holdouts. In
contrast, the directly connected *approach-assigned* MBONs 10599 and 508595
have 8 and 10 aversive DAN routes and showed reduced normal versus
no-plasticity spike counts in every holdout. That is a neural effect, **not**
proof of learned threat avoidance: the benchmark's behavioral claim remains
false, and the earlier 100-step reachable-arena benchmark had no successful
food holdout and an adverse threat comparison.

Next causal gate: record time-resolved KC excitation and other presynaptic
inhibition at the active, aversively modulated MBONs, then lesion their
registered descending route in a matched embodied holdout. Require a
specific normal-versus-lesion motor/body effect and positive food and threat
behavior across independent seeds before any autonomy claim. Do not increase
gains or route target coordinates into the controller to manufacture success.
