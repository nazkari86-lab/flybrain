# Typed descending-neuron lesion in the embodied MaleCNS loop

## Purpose and evidence boundary

Test whether the annotated DNg33 pair contributes to registered motor-neuron
spiking in the retained MaleCNS–FlyGym episode more than a similarly sized
descending pair, DNg48. This is a causal intervention in a simulation, not an
animal experiment or evidence for learned navigation. No motor decoder, gain,
learning rule, arena, or graph edge changes are permitted.

The types were selected before this intervention from the retained annotation
table and an earlier direct-MBON class diagnostic. DNg33 consists of cells
13317 and 13442; DNg48 of 12219 and 12465. Both pairs are annotated
`descending_neuron`, use acetylcholine in the canonical graph, and have 178
and 179 outgoing graph edges respectively. Their 16 versus 2 direct edges to
`vnc_motor` cells are an anatomical observation, **not** evidence of functional
necessity. The pairs are not matched for baseline firing, target identity, or
all synaptic features. Neither is in the hand-built six-population descending
motor decoder, avoiding its direct phase-modulation path in this comparison;
closed-loop sensory feedback can still differ after a lesion.

## Interface

Keep `descending_lesion="none" | "all_annotated"` behavior unchanged and add
`"annotated_type"` plus an optional exact `descending_type` string to the
episode, retained assay, and CLI (`--descending-type`). The string resolves
only cells for which `superclass == "descending_neuron"` and `cell_type` equals
the requested type. A provided type is monitored even with lesion `none`.
`annotated_type` requires a nonempty, existing type; `all_annotated` forbids a
type; an unknown or non-DN type fails closed. The output records the type,
its graph-cell count, its spike count, lesion mode, and silenced cell count.
The same selection and mask are used for exact replay. Default outputs are
otherwise behaviorally unchanged.

## Prospective experiment

After code and thresholds are committed, run seeds 13, 14, and 15 with FlyGym,
100 body steps, `--motor-trace`, and the same snapshot, registries, arena and
parameters as the prior all-DN study. For each seed, run unlesioned DNg33
monitor, DNg33 lesion, and DNg48 lesion. An unlesioned DNg48 monitor is run
once per seed to check that merely selecting a readout type does not alter the
baseline trace. No exploratory run on these seeds or post-hoc retuning is
allowed.

The primary endpoint is registered motor-neuron spike count. Declare a
*typed-DN differential* only if, in all three seeds, DNg33 baseline spikes
are positive, the DNg33 lesion has zero DNg33 spikes, both lesions replay
exactly without graph mutation, DNg33 lesion reduces motor spikes by at least
5% relative to its paired baseline, and its motor-spike reduction exceeds
the DNg48 reduction by at least 5 percentage points. These thresholds are a
fixed practical-effect screen, not an estimated population effect. Applied
moment, body displacement, food/threat contact, DAN events, and stability are
secondary diagnostics; they cannot rescue a failed primary gate. If any
criterion fails, report a negative or inconclusive result without selecting
another type from these same seeds.

Even a positive gate would establish only differential model-level influence
of two annotated DN types under this protocol. It would not prove a minimal
ORN→MBON→DN→muscle path, biological gait, learned food/threat behavior, or
real-fly validity.
