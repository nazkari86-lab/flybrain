# Locked design: cross-animal DNa02 relay-input rank in BANC

## Question and prior anchor

Does the female BANC v888 connectome reproduce the MaleCNS finding that
DNa02 ranks first among bilateral descending cell types by directed input
to the twelve ipsilateral IN19A003/IN08A006 cells? The previously published
MaleCNS audit selected these relays from literature and ranked DNa02 first
among 384 eligible bilateral descending types, with 12/12 relay coverage and
1,528 retained source synapses. This BANC analysis is an independent-animal
anatomical test, not a physiological or behavioral intervention.

## Source and selectors fixed before global-edge inspection

Use the same Harvard Dataverse deposit, version, and BANC materialization as
the earlier route audit: DOI `10.7910/DVN/7WTH1N`, version 3.0, v888.
The versioned metadata MD5 is `6275eda42f98c49539d1ab513d979d09`;
v2 edge MD5 is `394406f8a9bdf093c895f95aff4f6c49`; v3 edge MD5 is
`08542b0771db7418ed474be60dc9886c`. Use v2 as primary, v3 only as a
detector-version sensitivity check. In each edge table, a directed edge
qualifies at **at least five** synapses. Do not mix detector versions.

Eligible BANC source types must have `super_class == descending`,
`proofread == TRUE`, nonempty exact `cell_type`, and **exactly one** cell
with `side == left` and **exactly one** with `side == right` among all
proofread cells of that type. Metadata inspection before examining global
edges found 336 such types / 672 cells, including DNa02. Use the decimal
string `banc_888_id` as the join key. Exclude types with extra cells on
either side; do not collapse or cherry-pick their copies. The targets are
the same twelve proofread, type-exact IN19A003/IN08A006 relay cells in the
published BANC route audit: one per type, side, and T1/T2/T3 neuromere.

For each eligible source type, examine only directed source→relay edges
whose source and relay share a side. Define *coverage* as the number of
distinct relay cells with a qualifying edge, from zero to twelve. Define
*weight* as the sum of synapse counts on just these qualifying edges.
Rank types lexicographically by descending coverage, then descending
weight; break exact ties alphabetically by `cell_type`. The primary success
criterion is DNa02 rank **1**. Report its absolute coverage/weight, the
top ten types, the full number of tied 12/12 types, and the ranks of
DNg13 and DNge023 if eligible. Also report DNa02's left/right and relay-
type/segment counts rather than suppressing asymmetry. The pre-existing
BANC route audit's 12/12 and 822 v2 synapses are a consistency check.

Do not quote a hypothesis-test p-value for rank 1/336: cell types and
connections in one animal are not independent animal replicates, and the
relays were selected based on previous biological work. A positive outcome
supports conserved relative anatomical targeting across one male and one
female connectome. It cannot establish in-vivo synaptic sign, motor force,
stride change, causal specificity, or autonomous intelligence.

## Reproduction and failure handling

Verify each source MD5 before reading edges. Stream Feather record batches;
do not load a dense graph. Reject missing/duplicate selected root IDs,
duplicate pre/post edge pairs, incomplete relay side×segment inventory,
and invalid edge counts. Retain source hashes, exact eligible type list,
per-type coverage/weight, qualifying DNa02 edges, full ranking, software
revision, protocol identifier, threshold, and runtime in a JSON artifact.
Independently recompute the headline ranking from that artifact. Report a
negative or ambiguous outcome under these unchanged rules; do not change
the cohort, threshold, target set, or scoring after seeing the result.
