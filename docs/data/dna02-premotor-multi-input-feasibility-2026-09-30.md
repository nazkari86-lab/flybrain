# Multi-input same-post premotor control: prospective feasibility failure

The [predeclared design](../superpowers/specs/2026-09-30-premotor-matched-multi-input-control-design.md)
was committed as `67fd712` and the separate runner and tests as `fbf34b5`
before any seeds 46–48 were examined. The experiment stops at its declared
prior-panel balance gate. **No new-seed outcome was run.** This is not a
positive biological specificity result; it identifies why the earlier
under-exposed control cannot be repaired by a small number of alternative
same-post inhibitory edges under the fixed ORN/Shiu conditions.

## Source and frozen artifact

The canonical MaleCNS snapshot content SHA-256 is
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.
The registered ORN source JSON SHA-256 is
`3f3493acb4b61fa8cedb2f5cc8cb07c84b4367335161e2a9a7045c7c35f3864e`.
Only intact food and threat conditions on prior seeds 40–42 were used to
choose controls. The frozen
[control artifact](../../artifacts/dna02-premotor-multi-controls-prior40-42-fbf34b5.json)
has file SHA-256
`f972d13993f63e11160c5ad513f7e72f27769db172f6c757cddb52f49cd93920`
and selection digest
`0c9a415170906912e5f287981a548b242a896b48e944c21be56bda1fb142dfd3`.
It records exact edge IDs, CSR positions, weights, six prior-panel event
digests and modeled exposure vectors. Runtime was 11.23 s on this host.

Each of the 33 leg-rotator motor posts had at least five alternative
GABA-labelled `vnc_intrinsic` incoming edges in the retained graph (median
38, maximum 65). Exact enumeration over one to four edges per post chose
120 alternative edges: two posts received one, two received two, two received
three, and 27 received four. The reference route has 33 edges, one per post.

| ORN | Seed | Frozen alternative/route exposure |
| --- | ---: | ---: |
| Food | 40 | 0.5834 |
| Food | 41 | 0.7883 |
| Food | 42 | 0.8780 |
| Threat | 40 | 0.7632 |
| Threat | 41 | 0.9214 |
| Threat | 42 | 0.5002 |

Four of six panels miss the **unchanged 0.8–1.2** balance gate. The frozen
artifact reports `prior_balance_passed=false`. Invoking the holdout runner
with that artifact exits with code 2, `prior activity matching failed or was
altered`, and creates no outcome file. This refusal was checked directly.

## Exploratory constraint diagnostic, not a replacement protocol

Using only the same prior-panel source counts, an additional read-only
feasibility check formed a binary variable per one of 1,189 eligible
alternative edges. Each post had to select at least one and at most a tested
cap; all six *pooled* alternative/route exposure ratios had to lie in
0.8–1.2. SciPy 1.18.1 HiGHS MILP reported infeasible for caps 4 and 5.
At cap 6 it found a 152-edge solution with pooled ratios
`[0.802, 1.019, 1.178, 1.030, 1.200, 0.800]` in food-40–42 then
threat-40–42 order. This diagnostic uses more edges than the locked design,
does not match exposure per post or spike timing, and was **not** promoted
to an intervention. In particular, 152 alternative edges versus 33 route
edges would make a causal specificity comparison harder to interpret.

## Independent anatomical context and limits

[Stürner et al. (Nature 2025)](https://www.nature.com/articles/s41586-025-08925-z)
reported DNa02 output to IN08A006 and IN19A003 in MANC and FANC, and
published a [matched-neuron supplement](https://github.com/flyconnectome/2023neckconnective/blob/main/Supplemental_files/Supplemental_file13_other_MANC_FANC_matching.tsv).
Direct inspection of that supplement found FANC matches for all six
IN08A006 cells and five of six IN19A003 cells; the right-T2 IN19A003 is
explicitly marked `missing FANC match`. The downloaded publisher supplement
ZIP had SHA-256
`9f7b1193f3eecccde8cbdb35137a00cd4577d0420bf656a78d6829eeb11047fb`.
These are independent *published anatomical* observations, not a new
FlyBrain physiology experiment; the matching table alone does not supply
the FANC relay-to-muscle synapses or a measured inhibitory sign.

## Reproduction and status

```bash
uv run python -m flybrain.premotor_multi_control_probe freeze \
  artifacts/male-cns-v1.0-w5 \
  --odor-source-artifact artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz \
  --output /tmp/reproduced-premotor-multi-controls.json
```

For the source revision above, Ruff passed, mypy found no issues in 93
source files, and pytest reported 502 passed and 28 skipped. These checks
verify software behavior, not biological mechanism. The defensible
conclusion remains the published within-model pooled inhibitory effect;
the stronger activity-matched specificity, DNa02-dependent natural steering,
and living-fly motor physiology are unproven. Future work needs a genuinely
comparable active-input control or independent intervention/recording data,
not a relaxed threshold or a larger arbitrary control set.
