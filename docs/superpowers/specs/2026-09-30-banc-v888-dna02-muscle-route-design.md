# Locked design: independent BANC DNa02-to-rotator-muscle anatomy

## Question and scope

In the independent female BANC v888 connectome, are the literature-named
DNa02 → IN19A003/IN08A006 → leg-rotator motor routes present as directed
synapse-count edges to **motor neurons with explicitly annotated muscle
targets**? This is an anatomical replication and extension of the retained
MaleCNS motif, not a test of spike dynamics, postsynaptic inhibition, stride
change, or autonomous learning. The route types and mapping were specified in
the pre-existing MaleCNS report: IN19A003 → posterior rotator muscle and
IN08A006 → anterior rotator muscle. No BANC edge outcome may change those
identities or the threshold.

## Immutable sources

Use the Harvard Dataverse deposit DOI `10.7910/DVN/7WTH1N`, released version
3.0, BANC CAVE materialization v888. Verify before analysis:

| Source file | Dataverse file ID | MD5 |
| --- | ---: | --- |
| `banc_888_meta.feather` | 14033740 | `6275eda42f98c49539d1ab513d979d09` |
| `banc_888_edgelist_simple_v2.feather` | 13992792 | `394406f8a9bdf093c895f95aff4f6c49` |
| `banc_888_edgelist_simple_v3.feather` | 13918810 | `08542b0771db7418ed474be60dc9886c` |

Use v2 as the primary edge set because the BANC paper uses v2 throughout its
quantitative figures; use v3 as a sensitivity check. The v2 upstream synapse
size cutoff is 5 voxels, v3 is 10 voxels. In **both** edge tables call a
directed connection present when the edge count is at least **5**. Counts
must not be conflated across detector versions. The mutable public GCS
metadata file must not replace the versioned Dataverse file.

## Cell eligibility and matching

Take exactly the two `cell_type == DNa02` neurons, the two `DNg13` neurons as
a descriptive comparator, all six each of `IN19A003` and `IN08A006`, and
all neurons with `peripheral_target_type` equal to
`sternal_posterior_rotator_muscle` or
`sternal_anterior_rotator_muscle`. Require `proofread == "TRUE"` for every
eligible cell and `flow == "efferent"` for a motor target. Use the string
`banc_888_id` for edge joins; do not coerce 64-bit IDs to floating point.
Read side and neuromere from the metadata table. A relay→motor pair is
eligible in the primary analysis only when both side and *explicit*
neuromere match, with IN19A003 targeting posterior and IN08A006 anterior.
For a separately labelled sensitivity analysis only, infer a missing motor
neuromere from an unambiguous `root_region` token `_T1_`, `_T2_`, or `_T3_`
when its `_L`/`_R` token agrees with the `side` field. Do not silently include
ambiguous cells. DNa02→relay matching requires equal sides. DNg13 is checked
against only the same-side relays under the same threshold; it is not activity-
matched and cannot establish functional specificity.

## Outputs and verification

Report all eligible cell IDs and annotation fields, every observed route
edge (including subthreshold counts), the count and summed synapses of
threshold-passing DNa02→relay and relay→muscle-target edges, the six
side×neuromere coverage cells per relay type, and the DNg13 comparator.
Primary and root-region-inferred counts must remain separate. Retain the
source MD5/SHA-256, software revision, threshold, Dataverse DOI/version and
exact edge rows in a machine-readable artifact. Unit-test cell selection,
side/segment gating, subthreshold behavior, missing-segment sensitivity and
64-bit ID safety. Independently recalculate headline counts from the output.

A positive result can support conservation of *anatomical connectivity* and
an explicit route to annotated muscle-targeting MNs in another animal/sex.
It does not validate a transmitter's postsynaptic sign (predictions and
literature verification must be distinguished), muscle force, behavior,
learning, or causal necessity in living flies. Missing or asymmetric edges
must be reported without changing cell selectors or thresholds.
