# DNa02 same-postsynaptic controls: strict specificity gate failed

This is a **negative, prospectively locked model-circuit result** on the
retained MaleCNS graph. It does not erase the [previous narrow direct-edge
effect](dna02-registered-orn-direct-edge-2026-09-30.md), but it prevents a
stronger claim that DNa02's 17 edges have a selective effect beyond other
active positive inputs onto the same thoracic motor cells. No living fly,
real odorant, muscle, body, learning, or autonomous behavior was tested.

The [protocol](../superpowers/specs/2026-09-30-dna02-same-post-control-design.md)
was published as `becb0eccc9522848c5a365e60073c634b631f65e`; the
[runner](../../src/flybrain/dna02_same_post_probe.py) was published as
`167d25a0d23dc4399d3f0be1d8e6d72c60d5f946` **before** the previously
unused ORN seeds 40–42 were run. All thresholds were retained after seeing
the outcomes. The frozen control lists were selected with intact seed-37-to-39
food/threat spike counts from the preceding published assay, then checked
independently against the specified minimization rule: 17/17 single-edge
choices and 17/17 two-edge choices matched. Selection was therefore informed
by earlier model outcomes; the new-seed panel tests that selected contrast.

## Intervention and observations

For each food-labeled and threat-labeled registered ORN input, one Poisson
event dictionary was reused across intact, direct DNa02 17-edge-zero,
single same-post `vnc_intrinsic` 17-edge-zero, pair same-post `vnc_intrinsic`
34-edge-zero, no-source, and replay conditions. All ran 2,000 Shiu steps
(200 ms), with intact DNa02 afferents and no direct DNa02 stimulation. The
controls ended on the **same 17 annotated thoracic `vnc_motor` cells** as
the direct edges. Their presynaptic class and edge counts differ from DNa02;
the modeled weight×spike exposure was an approximate selection criterion,
not a measure of biological synaptic efficacy.

| ORN input | Seed | Target spikes intact → direct → single → pair | Target reduction: direct / single / pair | Pair input exposure ÷ direct | Strict gate |
| --- | ---: | ---: | ---: | ---: | :---: |
| Food-labeled | 40 | 99 → 87 → 86 → 84 | 12.12% / 13.13% / 15.15% | 0.766 | fail |
| Food-labeled | 41 | 140 → 127 → 125 → 124 | 9.29% / 10.71% / 11.43% | 1.272 | fail |
| Food-labeled | 42 | 135 → 116 → 122 → 121 | 14.07% / 9.63% / 10.37% | 0.943 | fail |
| Threat-labeled | 40 | 125 → 109 → 114 → 113 | 12.80% / 8.80% / 9.60% | 1.091 | fail |
| Threat-labeled | 41 | 95 → 92 → 86 → 85 | 3.16% / 9.47% / 10.53% | 1.436 | fail |
| Threat-labeled | 42 | 95 → 81 → 82 → 81 | 14.74% / 13.68% / 14.74% | 0.796 | fail |

Independent read-only recalculation from the complete JSON, without trusting
stored gate flags, confirmed: source ORN-event pairing **6/6**; exact DNa02
and selected control-source spike-time pairing **6/6**; unchanged aggregate
other-motor spikes **6/6**; DNa02 and target recruitment above no-source
**6/6**; no-source quiet, exact replay, and unchanged canonical graph
**6/6**. The direct lesion reached the minimum 5% target reduction in **5/6**.
It exceeded **both** same-post controls by the required 5 percentage points
in **0/6**. The two-edge control's weight×spike input exposure fell within
the locked 80–120% range in only **2/6**. In those two exposure-balanced
conditions (food 42 and threat 40), the direct effect exceeded the two-edge
control by only 3.70 and 3.20 percentage points, below the 5-point threshold.

This removes the earlier **source-feedback confound**: all monitored
presynaptic spike-time digests were equal in each paired intervention. It
does **not** establish a selective DNa02 effect. The approximate input match
failed four held-out conditions, and target-spike effects of other active
inputs were often as large or larger. The original narrow causal observation
that DNa02 direct-edge deletion changes these targets under ORN drive still
stands, but should not be promoted to unique physiological control, muscle
force, movement, learning, or autonomous intelligence. The previous
closed-loop DNa02 gate also remains failed 0/3.

## Artifact, reproduction, and next inference boundary

The [full compressed JSON](../../artifacts/dna02-same-post-control-seeds40-42.json.gz)
records every frozen edge identity, CSR position, signed model weight, source
and target readout, all condition digests, thresholds, software and graph
hashes, panel runtime (60.1 s, excluding graph loading), and peak RSS
(1.02 GB). `gzip -t` passed. Its **uncompressed** SHA-256 is
`c4749ca2086fb666ed2c8f6c88299854349105f23309e22db10b4274da3ebc72`.
The snapshot content hash remains
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.

```bash
gzip -dc artifacts/dna02-same-post-control-seeds40-42.json.gz | shasum -a 256
.venv/bin/python -m flybrain.dna02_same_post_probe \
  artifacts/male-cns-v1.0-w5 \
  --frozen-edge-artifact artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz \
  --odor-source-artifact artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz \
  --output artifacts/reproduced-dna02-same-post.json \
  --steps 2000 --seeds 40 41 42
```

Before the held-out run, the source revision passed the full local suite
(`496 passed, 28 skipped`), Ruff, mypy, a real-snapshot 20-step source/edge
validation, and a three-file CodeRabbit review with zero remaining findings.
These verify implementation integrity, **not** the model's correspondence to
a real fly. A stronger biological claim now needs an independently specified
route with target-level physiology and behavioral or muscle-output evidence,
not a post-hoc relaxed margin on these same six outcomes.
