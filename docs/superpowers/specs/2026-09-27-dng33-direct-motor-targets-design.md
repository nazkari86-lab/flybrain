# DNg33 direct motor-target spike readout in embodied MaleCNS

## Question and prior evidence

The prospective [typed-DN experiment](../../data/typed-descending-lesion-2026-09-27.md)
failed its registered-motor criterion. It exposed a measurement gap: both
annotated DNg33 cells (13317, 13442) connect to the same eight `vnc_motor`
cells (800659, 803732, 810086, 813291, 814430, 814989, 815205, 815281),
none of which is in the 182-cell, 36-population motor registry. The retained
graph has 16 DNg33→target edges with combined signed weight +404. This
anatomical observation is not functional evidence.

Test whether DNg33 spiking causally influences those eight annotated motor
targets in the existing MaleCNS–FlyGym closed loop. Do not route the eight
cells into the torque decoder, alter synaptic weights, change gains, move
objects, or introduce a planner/RL policy.

## Readout-only interface

Add `spike_readout_superclass="none" | "vnc_motor"` to the episode, retained
assay, and CLI (`--spike-readout-superclass`). With `vnc_motor`, resolve all
graph neurons whose superclass is exactly `vnc_motor`; fail closed if none.
Record the selected superclass and a per-neuron spike-count dictionary that
includes zero-spike cells. Counts come from the existing emitted Shiu neuron
IDs, before motor decoding. The readout must not enter neural dynamics, the
body, rewards, or motor actions. Default `none` keeps the original behavior.
Exact replay must include the readout counts. The existing typed-DN silence
mask and no-autonomy-claim gate remain unchanged.

## Locked prospective comparison

After the interface and tests are committed, run FlyGym 2.1.0 at 100 body
steps, the same snapshot/registries/arena/parameters, seeds **16, 17, 18**.
Each seed has three conditions: DNg33 monitored but intact (`descending_type`
`DNg33`, `descending_lesion=none`), DNg33 type lesion, and DNg48 type lesion.
All record the full `vnc_motor` readout and motor trace. The eight target IDs
above are fixed before these runs. Every other `vnc_motor` cell is a
predefined aggregate comparison group, not an activity- or anatomy-matched
control; the DNg48 pair was previously silent and is only a no-effect
intervention reference.

The primary endpoint is total spikes in the eight direct targets. Declare a
positive *model-level direct-target influence* only if in **every** seed:

1. the intact target sum and intact non-target sum are positive, and the
   DNg33 lesion emits zero DNg33 spikes;
2. DNg33 lesion reduces target spikes by at least 20% relative to the
   matched intact episode;
3. that target reduction exceeds the percent reduction in all other
   `vnc_motor` cells by at least 10 percentage points;
4. all conditions replay exactly, leave the canonical graph unchanged, and
   preserve the same source revision, snapshot and registry digests.

The DNg48 intervention, applied moment, body displacement, contacts, DAN
events, and stability are secondary diagnostics; none may rescue a failed
primary gate. Report a null or inconclusive result if any criterion fails,
without changing the eight targets, thresholds, horizon, seeds, or decoder.

Even if positive, this closed-loop lesion plus direct anatomical edges cannot
prove the effect travels only through those synapses: polysynaptic and
body-feedback routes remain. It cannot prove muscle force, biological gait,
learned food/threat behavior, or validity in a living fly. A fixed-source
neural replay with target-specific synapse intervention would be the next
localization gate.
