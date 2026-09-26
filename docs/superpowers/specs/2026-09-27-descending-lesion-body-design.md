# Annotated descending-neuron lesion in the embodied MaleCNS loop

## Purpose

Test whether the naturally recruited, annotated descending-neuron (DN) layer is
necessary for neural motor output and body motion in the existing retained
MaleCNS–FlyGym episode. The isolated-graph ORN experiment in
`docs/data/natural-odor-descending-lesion-2026-09-27.md` found zero registered
motor spikes after all-DN silencing in six paired conditions, but connected no
body and used no learning.

## Choice and alternatives

The selected approach is an explicit experimental lesion within the existing
Shiu silence mask. Replacing the decoder with a hand-built gait controller
would not test the connectome. Re-running an unmodified long behavioral
benchmark would not isolate the DN bottleneck. No new planner, target-bearing
action, gain adjustment, or anatomical edge is introduced.

## Interface

`run_autonomous_hexapod_episode` gains `descending_lesion="none" |
"all_annotated"`; the retained assay and `autonomous-hexapod` CLI expose the
same option. `none` is the default. `all_annotated` selects every neuron whose
retained graph superclass is exactly `descending_neuron`, regardless of which
six named DN populations the decoder exposes. An empty annotated selection
fails closed. The result records the selected mode and neuron count. Replay
runs with the same mask. The canonical graph, plastic overlay, learning rule,
motor decoder, environment, and body parameters are unchanged.

## Evidence and failure handling

A synthetic chain with one annotated DN must show nonzero motor output in the
baseline and zero motor output after DN silencing, with exact replay and an
unchanged graph. The retained test must show the serialized lesion identity,
positive annotated-neuron count, exact replay, and zero registered motor
spikes. A matched normal/lesion FlyGym pair uses the same seed, horizon, arena,
and backend; compare actual thorax displacement and joint moments against the
passive all-motor lesion. If the normal trajectory is indistinguishable from
passive settling, the embodied causal result is null, not positive.

This remains a model-level result. It does not permit claims of learned food
seeking, threat avoidance, a validated neuron-to-muscle map, or complete
biological intelligence.
