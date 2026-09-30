# Literature-anchored DNa02 premotor motif in retained MaleCNS

This is a **descriptive connectome audit**, not a new discovery about living flies or a
positive autonomous-behavior assay. It identifies a biologically interpretable route
that the project had not previously quantified: DNa02 → two serial GABAergic
premotor cell types → ipsilateral leg-rotator motor neurons. The route was selected
from published work before its edges were inspected in the retained snapshot.
The background ranking below was exploratory; no inferential threshold, independent
holdout, or source-locked prospective outcome test was performed.

## Independent biological anchors

- [Yang et al., *Cell* 2024](https://doi.org/10.1016/j.cell.2024.08.033)
  measured ipsiversive turning and shorter ipsilateral strides after unilateral
  DNa02 activation in walking flies. This is experimental physiology, not a
  validation of FlyBrain's neural dynamics or motor decoder.
- [Cheong et al., *eLife* 2026](https://doi.org/10.7554/eLife.96084)
  described DNa02 output to GABAergic IN19A003 and IN08A006 serial sets,
  which connect to leg-rotation motor neurons; it also described a different
  DNg13 premotor route. The source is a MANC connectome analysis, not an
  independent manipulation of these exact synapses.
- [Stürner et al., *Nature* 2025](https://doi.org/10.1038/s41586-025-08925-z)
  compared MANC and FANC and found DNa02's major downstream partner types,
  including IN19A003 and IN08A006, largely conserved across these datasets.
  Absolute synapse counts differed, so cross-dataset numerical equality is
  not expected.

## Retained-data method and provenance

Read-only PyArrow filters were applied to
`artifacts/male-cns-v1.0-w5/source-annotations.parquet`,
`neurons.parquet`, and `edges.parquet`. The snapshot content SHA-256 was
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`;
the checkout was `c9798623cf25f470c8a8ea305542120b3881f8e5`. The import
keeps only directed edges with at least five source synapses. The specified
cells were type-exact DNa02, DNg13, IN19A003, IN08A006, Sternal anterior
rotator MN, and Sternal posterior rotator MN. No simulator run, new parameter,
post-hoc edge replacement, or dense connectome allocation was involved.

For the exploratory background, include every `descending_neuron` type with
exactly one annotated left and one right soma-side cell; 384 types qualify.
Coverage is the number of the 12 specified relay cells receiving a retained
edge from the DN cell on the *same soma side*. Ties in coverage are ordered
by summed source synapse count into those relays. This comparator does not
match presynaptic activity or establish functional specificity.

## Observations

| Stage | Retained result |
| --- | ---: |
| DNa02 → IN19A003/IN08A006 | 12/12 ipsilateral relay cells; 12 edges, 1,528 source synapses |
| DNg13 → the same relays | 0 retained edges |
| IN19A003 → Sternal posterior rotator MN | 21 edges, 5,301 source synapses |
| IN08A006 → Sternal anterior rotator MN | 12 edges, 3,642 source synapses |
| Combined relay → rotator MN | 33 inhibitory-signed edges to 33 of 34 typed rotator MNs; 8,943 source synapses |

Both DNa02 copies reach all six ipsilateral relay cells: one IN19A003 and
one IN08A006 in each of T1, T2, and T3. All 12 relays are annotated
`Reviewed`; both DNa02 source cells are `Prelim Roughly traced`. The 33
relay-to-motor edges are ipsilateral and connect cells in the same thoracic
neuromere. Their 33 postsynaptic motor cells are all `Reviewed`. There are
18 such edges on the left and 15 on the right; this audit does not claim
perfect left-right numerical symmetry.

The retained neuron table labels both DNa02 copies `acetylcholine` and all
12 relay cells `gaba` under `male-cns-consensus`. Corresponding graph signs
are +1 for DNa02 → relay and −1 for relay → motor, each with
`fast-transmitter-assumption` provenance. This is a transmitter-based sign
model, **not** measured postsynaptic receptor physiology.

Among the 384 eligible bilateral DN types, only DNa02 and DNge023 reached
all 12 specified relays; their summed retained input weights were 1,528
and 792 respectively. Thus DNa02 ranks first under the declared descriptive
coverage/weight ordering, but the route is not unique. DNg13's absence from
these *specific* relays is consistent with the published distinction between
the two premotor pathways; it does not imply DNg13 lacks leg-motor effects.

## Claim boundary and next causal test

This establishes that the retained MaleCNS **contains a literature-identified,
bilateral cholinergic-to-GABAergic-to-leg-motor motif** under its import and
annotation rules. It does not establish relay firing under natural sensory
input, inhibitory postsynaptic potentials, muscle force, phase-specific stride
shortening, steering, learning, or autonomous intelligence. The previously
published DNa02 same-postsynaptic specificity gate still failed 0/6; this
anatomical finding does not reverse it.

The next stronger result would require a source-locked functional protocol:
same registered sensory events in each arm; measured DNa02 and relay spike
timing; selective, ipsilateral relay-output interventions with active-input
same-post controls; identical non-target motor activity checks; and an
independently justified motor-to-muscle/stride readout. A result on this
snapshot would still be model-level until compared against experimental
physiology. No new positive causal or behavioral claim is made here.
