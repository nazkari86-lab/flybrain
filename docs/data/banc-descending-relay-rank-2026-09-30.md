# DNa02 relay-input rank across male and female connectomes

This is a new **comparative anatomical** result for FlyBrain. The female BANC
v888 animal reproduces the earlier MaleCNS observation that DNa02 ranks first
among bilateral descending types by input to the twelve same-side
IN19A003/IN08A006 premotor relay cells. In BANC, DNa02 is the **only one of
336 eligible types** with qualifying input to all twelve. This is not a
new intervention in a living fly, a measurement of muscle force, or proof of
autonomous intelligence.

## Locked question and provenance

The [design](../superpowers/specs/2026-09-30-banc-descending-relay-rank-design.md)
was committed at `4de4fb26e3631f0c342a7bb7e13fa459220cee66` before the
global BANC descending→relay edges were inspected. The
[auditor](../../src/flybrain/banc_dn_rank_audit.py) and
[synthetic tests](../../tests/test_banc_dn_rank_audit.py) were committed at
`25e644df1da0f656e0584a5b492a62027a74afc2` before the real-data run.
The criterion was fixed as lexicographic descending (number of the twelve
relays receiving at least five directed same-side synapses, sum of the
qualifying synapse counts), then type name to break exact ties. It includes
only proofread BANC `super_class == descending` types with exactly one
left and one right cell. The relays were selected from published biology
in the earlier MaleCNS analysis, not discovered by this rank test.

The source is [Harvard Dataverse DOI 10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N),
deposit version 3.0, BANC materialization v888. The versioned metadata,
v2 edge, and v3 edge files have source MD5 values
`6275eda42f98c49539d1ab513d979d09`,
`394406f8a9bdf093c895f95aff4f6c49`, and
`08542b0771db7418ed474be60dc9886c`. They were checked before analysis.
The full [219 KB result](../../artifacts/banc-dn-relay-rank-v888-25e644d.json)
has SHA-256 `7fd5c075131164d130e4ab9a6edf3ef773bfcc82d9821b6f32251bc03120c0bd`.
It contains the 672 eligible source cells, twelve relays, complete ranking,
all qualifying DNa02 edges, source SHA-256 values, and revision metadata.
The large source Feather files are not checked into Git.

## Prespecified primary result

| Female BANC type | v2 relay coverage | v2 qualifying synapses | v3 sensitivity coverage / synapses |
| --- | ---: | ---: | ---: |
| DNa02 | **12/12** | **822** | **12/12 / 946** |
| DNge006 | 9/12 | 454 | 9/12 / 490 |
| DNge023 | 8/12 | 271 | 8/12 / 306 |
| DNg13 | 0/12 | 0 | 0/12 / 0 |

DNa02 is rank 1/336 in both detector versions and the sole type at 12/12.
In the v2 primary data it contributes 435 synapses from the left cell and
387 from the right; 541 target IN08A006 and 281 target IN19A003. All six
side×segment positions per relay type have a qualifying edge. The next
type reaches nine relays, not twelve. The v2 DNa02 total exactly matches
the previously published BANC route audit's 822, which serves as an
internal consistency check rather than a second independent animal.

The prior [MaleCNS anatomy audit](dna02-literature-anchored-premotor-motif-2026-09-30.md)
ranked DNa02 first among 384 bilateral descending types by the analogous
coverage/weight ordering (12/12, 1,528 source synapses). MaleCNS also had
DNge023 at 12/12; BANC has it at 8/12. This supports *relative anatomical
prominence* of DNa02 in two animals of different sexes. Because each
connectome comes from one animal, the numeric weight difference and the
DNge023 coverage difference are not population or sex-effect estimates.

## Verification and claim boundary

An independent read-only PyArrow scan of the three versioned source files,
separate from the auditor code path, reproduced the v2 top three as
`(DNa02,12,822)`, `(DNge006,9,454)`, `(DNge023,8,271)` and the v3 top
three as `(DNa02,12,946)`, `(DNge006,9,490)`, `(DNge023,8,306)`.
The only full-coverage type was DNa02 in both scans. Ruff, mypy and the
full pytest suite passed before the first real-data run: 509 passed,
28 skipped. These checks support the analysis implementation, not a
physiological mechanism.

The DNa02 BANC edge counts and complete 12/12 coverage were already known
from the previous route audit when this ranking was designed. The new
held-out quantity is the **relative rank against every eligible descending
type**, not the existence of those twelve edges. The relay target set was
chosen because of prior DNa02 biology, so the 1/336 rank is descriptive;
it is not a hypothesis-test p-value or evidence that 336 neuron types are
independent animal replicates. v3 is a detector-version check on the same
animal, not biological replication.

This finding does not determine postsynaptic sign in vivo, whether the
selected relays are causally necessary for DNa02-driven stride shortening,
actual rotator muscle output, or learned autonomous behavior. The existing
same-post functional specificity and autonomous-behavior gates remain
failed. A stronger biological bridge needs relay-specific perturbation or
simultaneous DNa02→relay→motor/muscle measurement in living walking flies,
with matched controls.

Reproduce the source-locked calculation after obtaining the three Dataverse
files at the stated MD5 values:

```bash
uv run python -m flybrain.banc_dn_rank_audit \
  --metadata artifacts/banc-v888-external/banc_888_meta.feather \
  --edges-v2 artifacts/banc-v888-external/banc_888_edgelist_simple_v2.feather \
  --edges-v3 artifacts/banc-v888-external/banc_888_edgelist_simple_v3.feather \
  --output /tmp/banc-dn-relay-rank-recheck.json
```
