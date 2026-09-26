# Registered olfactory input to descending and motor recruitment — exploratory lesion check

This is an isolated *simulation* result on the retained MaleCNS graph, not an
embodied, learned-navigation, or complete-intelligence result. It extends the
[odor-to-MBON bottleneck probe](odor-to-mbon-bottleneck-2026-09-26.md) with
readouts from all annotated descending neurons (DNs) and registered motor
neurons. The source checkout was `2af307ac55bd539abb4ddb5ad7c76a51226950bc`.
The retained graph has 166,606 neurons and 6,240,402 edges; its previously
recorded graph digest is
`0603c4a7b351d21d9ede778478e5a15f746b7b08747f1c43654c1b05d9530404`.

## Protocol

Both-side ORN_DM1/ORN_VA2 populations were the food-labeled source; both-side
ORN_DA2 was the threat-labeled source. These are task-assigned odor channels,
not delivered or measured real odorants. Each source used the existing
`OlfactoryReceptorMap`, `task_odor_assignment`, and `poisson_voltage_events` at
the default 150 Hz source-equivalent rate. Each condition ran 2,000 Shiu steps
(`dt_ms=0.1`, `refractory_ms=2.0`, `synaptic_delay_ms=1.0`). The KC→MBON overlay
stayed at unity: **there was no training or reward in this assay**.

Seeds 7, 8, and 9 were run separately. For each seed and stimulus, the exact
same pre-generated source-event dictionary was reused across conditions, with
a fresh `ShiuState.initial` and the same seed. The conditions were no lesion;
silencing MBONs 10599 and 508595; silencing the existing transmitter/outdegree
matched controls 11176 and 12726; silencing registered left and right DNg13;
and silencing all 1,314 neurons with retained `descending_neuron` superclass.
The matched MBONs were **not matched for baseline activity**. The no-source
control was run for each seed without a lesion.

This was designed after seeing the earlier MBON recruitment and 100-step
behavior results, so the seed-7-to-9 panel is exploratory, not preregistered.
It does not estimate a population-level effect or confidence interval. The
normal seed-7 food and threat neural trace digests exactly matched the earlier
published odor probe (`6c1ddd375e27e7231115e56bbc35b7ae48e3890fbcbe7078877199d6f6888857`
and `dd03ed29b9f8db129a9dfa5a9d674d804552793ad5e64d7641743e9aea299d30`,
respectively). Other lesion traces were not independently replay-digested.

## Observations

| Stimulus | Seed | Source voltage events | Normal all-DN spikes | Normal motor spikes | Motor spikes with 10599+508595 silent | Motor spikes with matched MBONs silent | Motor spikes with DNg13 silent | Motor spikes with all DNs silent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 7 | 0 | 0 | 0 | — | — | — | — |
| None | 8 | 0 | 0 | 0 | — | — | — | — |
| None | 9 | 0 | 0 | 0 | — | — | — | — |
| Food | 7 | 4,288 | 4,040 | 266 | 260 | 264 | 251 | 0 |
| Food | 8 | 4,356 | 4,522 | 274 | 318 | 289 | 256 | 0 |
| Food | 9 | 4,450 | 4,578 | 259 | 263 | 259 | 275 | 0 |
| Threat | 7 | 1,209 | 3,806 | 233 | 246 | 223 | 215 | 0 |
| Threat | 8 | 1,190 | 3,530 | 206 | 195 | 199 | 198 | 0 |
| Threat | 9 | 1,192 | 4,225 | 272 | 253 | 254 | 240 | 0 |

Food-labeled stimulation evoked MBON 10599 counts of 20/20/21 and MBON
508595 counts of 21/21/21 across seeds 7/8/9. Threat-labeled stimulation
evoked 10599 counts of 18/16/17 and 508595 counts of 15/17/22. MBON 519128
remained silent in every food and threat normal condition. All three no-source
conditions yielded zero total spikes in the graph.

The strongest new model-level causal result is broad: under registered ORN
population stimulation, all-DN silencing abolished registered motor spikes in **all
six paired odor/seed conditions**, whereas each unlesioned condition recruited
more than 3,500 DN spikes and 200 motor spikes. This is evidence for an
odor-input → annotated-DN → registered-motor dependency in this Shiu model.
It is **not** evidence for one specific MBON→DN route: removing the two active
MBONs changed motor counts in both directions across seeds, and the DNg13
lesion did not consistently reduce food-evoked motor spikes. Neither odor
produced a clean, MBON-specific valence signature. The broad all-DN lesion
could interrupt multiple parallel pathways, so this experiment does not
identify the minimal circuit.

No body was connected, no movement or contact was measured, and no plasticity
or post-training holdout occurred. Therefore this result must not be used to
claim food seeking, threat avoidance, learned behavior, or biological
intelligence. The next causal gate is a retained-body experiment with the same
source events and isolated DN lesions, plus replay and motor/body readouts;
after that, test learning on independent, unreused seeds against no-plasticity,
DAN, KC→MBON, and structural-rewire controls.
