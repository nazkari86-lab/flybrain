# Independent BANC DNa02-to-leg-rotator anatomy

This is a positive **connectome-anatomy** result in an independent female
BANC animal, not a measurement of living-fly physiology or autonomous
intelligence. The [protocol](../superpowers/specs/2026-09-30-banc-v888-dna02-muscle-route-design.md)
was committed as `59a6b9a` before inspecting the BANC edges. The auditor was
committed as `6872094`. It used only type-exact, fully proofread cells, directed
edges with at least five synapses, same-side DNa02→relay pairs, and same-side,
same-neuromere relay→motor pairs. The motor cells have explicit peripheral
leg-rotator muscle-target annotations. v2 is primary; v3 is a detector-version
sensitivity check, not another animal.

## Sources and exact result

The source is [Harvard Dataverse DOI 10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N),
deposit version 3.0, BANC materialization v888. Source-file MD5 values checked
by the auditor are `6275eda42f98c49539d1ab513d979d09` (metadata),
`394406f8a9bdf093c895f95aff4f6c49` (v2 edges), and
`08542b0771db7418ed474be60dc9886c` (v3 edges). Their SHA-256 values,
file sizes, selected cell IDs and annotation fields, and every relevant edge
are retained in the [94 KB result](../../artifacts/banc-dna02-rotator-v888-6872094.json)
(SHA-256 `cf221c4a5efeb7a97cb5b198a052242247ee20fc45ec74f96d107836b2c692bb`).
The large Feather sources are not checked into Git.

| Predeclared observation | v2 primary | v3 sensitivity |
| --- | ---: | ---: |
| DNa02→same-side IN19A003/IN08A006 | 12/12 relay cells, 822 synapses | 12/12, 946 |
| DNg13→those same relays | 0 qualifying edges | 0 |
| Relay→annotated rotator-target motor cells, explicit segment | 31 edges, 7,683 synapses; 11/11 eligible side×segment slots | 31, 7,960; 11/11 |
| Separately including one motor segment inferred from `root_region` | 34 edges, 8,543; 12/12 slots | 34, 8,866; 12/12 |

The missing explicit-segment slot is left T3 posterior rotator. Its motor
neuromere field is null; the separately labelled sensitivity analysis infers
T3 from the side-consistent `root_region`. The passing edge identities are
unchanged between v2 and v3. Both DNa02 cells have verified acetylcholine
annotations; six IN19A003 have verified GABA annotations, whereas GABA for
the six IN08A006 cells is **predicted, not verified**. These labels do not
measure postsynaptic sign or muscle force. DNg13 is a descriptive comparator,
not an activity-matched causal control.

## Verification and interpretation

On the committed auditor revision, regenerating the JSON from all three
Feather files yielded a byte-identical result. A separate read-only PyArrow
calculation from the source files reproduced the 12/0/31/34 edge counts and
their v2/v3 synapse sums. The project's Ruff, mypy, and full pytest checks
reported no errors, no type issues in 94 source files, and 506 passed / 28
skipped. These checks establish reproducibility of this analysis, not
biological function.

To reproduce after obtaining the three versioned Dataverse files:

```bash
uv run python -m flybrain.banc_route_audit \
  --metadata artifacts/banc-v888-external/banc_888_meta.feather \
  --edges-v2 artifacts/banc-v888-external/banc_888_edgelist_simple_v2.feather \
  --edges-v3 artifacts/banc-v888-external/banc_888_edgelist_simple_v3.feather \
  --output /tmp/banc-dna02-rotator-recheck.json
shasum -a 256 /tmp/banc-dna02-rotator-recheck.json
```

The defensible claim is that the selected DNa02→premotor→annotated
leg-rotator-motor route is present under the locked BANC v888 anatomical
rules, independently of the retained MaleCNS animal. It does **not** prove
synaptic sign in vivo, actual muscle activation, stride change, steering,
learning, or a complete autonomous fly brain. The earlier same-postsynaptic
functional control failed; this anatomical result does not reverse it.
Those remaining claims require independent recording or perturbation in
living flies and a validated motor-to-muscle/behavior mapping, with
prespecified matched controls.
