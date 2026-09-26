# Foreleg proprioceptive-subtype recruitment — 2026-09-26

This is a diagnostic of the retained MaleCNS/Shiu graph, **not** evidence of
walking, learning, or a complete autonomous fly. It tests whether annotated
foreleg sensory subsets can recruit registered motor neurons when stimulated
directly. It does not test the present body-to-sensor encoder, which still
averages eight proprioceptive channels for each entire leg bank.

## Reproduction and provenance

Use `flybrain experiment foreleg-subtypes` as documented in the README. The
published 500-step run is
[`artifacts/foreleg-subtypes-malecns-seed7-steps500-v2.json`](../../artifacts/foreleg-subtypes-malecns-seed7-steps500-v2.json).
It uses the retained `male-cns-v1.0-essential` graph (166,606 neurons;
6,240,402 directed edges), registry `hexapod-motor-registry-v2`, Shiu defaults
(`dt=0.1 ms`), seed 7, and one 10-mV pulse per selected neuron every 25 steps
for 500 steps (50 ms). Exact input IDs, graph/annotation/registry hashes,
per-population motor counts, event digests, and trace digests are in the JSON.

The protocol stimulates all neurons of one annotated subclass at a time.
Each condition is repeated, then run with that source subset silenced while
the input event schedule stays the same. It also runs homologous subclasses
on both sides. The right foreleg has no annotated hair-plate cell in this
registry, so no corresponding right control is claimed. Subclass sizes are
unequal; this is **not** an equal-input comparison of biological efficacy.

## Measured results

| Source side and subclass | Cells | Source spikes | Registered motor spikes | Foreleg tibia response |
| --- | ---: | ---: | ---: | --- |
| Left campaniform sensilla | 2 | 40 | 0 | none |
| Left chordotonal organ | 20 | 400 | 15 | 12 left flexor |
| Left hair plate | 1 | 20 | 0 | none |
| Left `leg` | 18 | 360 | 14 | 4 left extensor |
| Right campaniform sensilla | 2 | 40 | 0 | none |
| Right chordotonal organ | 10 | 200 | 7 | none |
| Right `leg` | 5 | 100 | 0 | none |

The other three left-chordotonal motor spikes are in the left-middle
trochanter flexor. The other ten left-`leg` spikes are spread over six
registered populations, including left and right thorax–coxa groups. The
seven right-chordotonal motor spikes occur in **right-middle** tibia flexor
(1) and trochanter flexor (6), not right-fore tibia. Thus `right motor
spikes > 0` is not a right-fore tibia success.

Every source-silenced condition has zero source and motor spikes; its event
digest matches the corresponding unsilenced condition. All seven exact
replays match their normal trace digests, and the graph digest is unchanged.
These are simulation-level causal controls only. They do not establish
natural sensory coding, realistic muscle activation, gait, or learning.

The [180-step baseline](../../artifacts/foreleg-subtypes-malecns-seed7-steps180.json)
gave 2 left-fore flexor spikes from left chordotonal cells and 1 left-fore
extensor spike from the left `leg` group, with no right-fore tibia spikes.
A [30-mV amplitude control](../../artifacts/foreleg-subtypes-malecns-seed7-steps180-amp30.json)
gave the same spike counts, so raising these already-suprathreshold pulses
did not fix right-fore recruitment. A [10-step interval control](../../artifacts/foreleg-subtypes-malecns-seed7-steps180-interval10.json)
gave the same motor counts with fewer source spikes (six rather than eight
per cell over 180 steps), consistent with the declared 2.2-ms refractory
period. Neither variation makes the experiment a natural closed loop.

## Decision boundary

The left-fore flexor **has a reachable path** under this artificial
chordotonal intervention. Its weak activation in the previous body trace
therefore cannot be attributed simply to missing anatomical inputs. This
experiment does **not** establish that whole-bank averaging is the cause:
the body encoder, stimulus statistics, graph dynamics, motor decoder and
unmeasured proprioceptive modalities all remain possible contributors.

The next controlled implementation candidate is an annotation-aware
body-to-source encoder, explicitly labelled a model assumption. Compare it
with the existing equal-mean encoder under matched source-spike budgets,
identical body and task conditions, source/motor lesions, exact replay and
independent appetitive/aversive holdouts. Do not promote motor recruitment
alone to an autonomous-intelligence claim.
