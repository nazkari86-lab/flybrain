# ORN-driven DNa02-linked inhibitory premotor output: a narrow positive model result

The retained MaleCNS/Shiu model shows a reproducible **pooled inhibitory-output
effect** for the literature-identified IN19A003/IN08A006 → leg-rotator motor
route. Under registered, artificial ORN drive, zeroing the 33 route edges
increased spikes in their 33 annotated motor posts in all six new panels,
while observed DNa02 and all monitored presynaptic spike timing stayed
identical. This is a causal result **inside one simulator**, not a demonstration
of DNa02-dependent natural steering, individual-edge necessity, muscle force,
learned locomotion, or intelligence. The prospectively fixed **strict
same-post specificity gate failed 0/6** because its alternative input
exposure was too low; one panel also missed complete relay recruitment.

## Source lock, prior data, and anatomy

The [design](../superpowers/specs/2026-09-30-dna02-premotor-output-control-design.md)
was published as `f60c8363ce5498f7ad6b4ac0e435ede1bb6fbc36`; the
[runner](../../src/flybrain/premotor_output_probe.py) and tests as
`2841c9a7f71a01163990e53f7111b57f1c4a70f9`; and the
[frozen control map](../../artifacts/dna02-premotor-controls-prior40-42-2841c9a.json)
as `e5a8d7dbd58cdcc3b8c1227149cf20953e1a3e48`, **before** the
new seeds 43–45 were run. The snapshot content SHA-256 is
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.
The frozen control JSON SHA-256 is
`488ceb8b7b8d4af78e46f8ca966ec24da0409e49e85aed75b1192cc02e1f0295`;
its selection digest is
`bba06ad51463ddda2126df3631ad66e3c07280f4049170a9541cf215dfaae1e4`.

The [preceding annotation audit](dna02-literature-anchored-premotor-motif-2026-09-30.md)
specifies the exact 12 GABA-labelled relays, 33 inhibitory-signed route
edges, and 33 motor posts. The underlying biological interpretation is
anchored by [Yang et al. on DNa02-dependent leg gestures](https://doi.org/10.1016/j.cell.2024.08.033),
[Cheong et al. on these premotor cell types](https://doi.org/10.7554/eLife.96084),
and [Stürner et al. on cross-connectome partner conservation](https://doi.org/10.1038/s41586-025-08925-z).
These publications do not validate the FlyBrain model's synaptic gains or
ORN-to-muscle dynamics.

For each motor post, other GABA-labelled `vnc_intrinsic` inputs to the *same
cell* were selected using intact source-spike counts from food/threat ×
seeds 40–42 only. One-edge and two-edge controls minimized the declared
six-panel absolute weight×spike exposure error. Their identities and CSR
positions were frozen before new outcomes. Already on the prior panel, the
two-edge control had only 36–78% of the route's pooled modeled input
exposure. The locked 80–120% adequacy threshold was **not relaxed** after
seeing this mismatch or after the new panel.

## New-seed intervention and observations

Every seed used one 2,000-step ORN event dictionary for intact, route-zero,
single-control-zero, pair-control-zero, no-source, and intact-replay
conditions. Edge removal used sparse zero multipliers; there was no body,
motor decoder, plasticity, reward, or direct stimulation of DNa02 or the
relays. Model edge signs derive from transmitter assumptions, not measured
postsynaptic receptor effects.

| ORN | Seed | 33 motor spikes: intact → route zero → single → pair | Route increase | Pair exposure / route | Strict gate |
| --- | ---: | --- | ---: | ---: | :---: |
| Food | 43 | 140 → 247 → 182 → 211 | 76.43% | 57.03% | fail |
| Food | 44 | 132 → 235 → 176 → 200 | 78.03% | 63.90% | fail |
| Food | 45 | 144 → 266 → 185 → 203 | 84.72% | 60.85% | fail |
| Threat | 43 | 123 → 210 → 156 → 171 | 70.73% | 55.50% | fail |
| Threat | 44 | 108 → 235 → 134 → 143 | 117.59% | 38.75% | fail |
| Threat | 45 | 99 → 197 → 125 → 147 | 98.99% | 47.21% | fail |

Independent read-only recalculation from the full JSON, without accepting
its stored gate flags, confirmed identical ORN-event, DNa02-spike-time and
all-selected-presynaptic-spike-time digests **6/6**; identical aggregate
other-motor spikes **6/6**; no-source quiet, exact intact replay, and
unchanged graph **6/6**. The route increased pooled target activity by at
least 25% and by at least 10 percentage points more than both controls
**6/6**. But the pair-control modeled input-exposure ratio was only
0.387–0.639, failing the predeclared 0.8–1.2 balance criterion **6/6**.
All 12 relay cells fired in **5/6** intact panels; one relay was silent in
threat seed 43. Thus the *overall strict gate passed 0/6*. Because the
controls are under-exposed, the numerical route-versus-control margin is
not evidence that this pathway has a selective effect beyond equally
effective alternative inhibitory input.

A separate exploratory prior-seed diagnostic zeroed the 12 DNa02→relay
edges. Relay spikes fell 160→101 in food seed 40, but DNa02 firing changed
31→32, confounding a clean source-specific comparison. Five canonical
IN19A003→DNa02 inhibitory feedback edges were found; zeroing them in both
arms **did not** restore DNa02 spike-time equality. No claim that DNa02 is
uniquely necessary for relay recruitment follows from that diagnostic.

## Reproduction, integrity, and limit

The [complete compressed result](../../artifacts/dna02-premotor-output-holdout43-45-e5a8d7d.json.gz)
passed `gzip -t`. SHA-256 of its **uncompressed JSON** is
`a6bf0b9a6ccf7da48e2e85ee9a4b963f64fea303678386ee1a0d5a5a8637e728`.
The locked run recorded source revision `e5a8d7dbd58cdcc3b8c1227149cf20953e1a3e48`,
61.45 seconds runtime and 1.13 GB peak RSS on this host. Re-running both
seed-43 panels produced the exact same JSON-normalized per-panel results.
The full local suite before the holdout was `499 passed, 28 skipped`;
Ruff, mypy and a two-file CodeRabbit review passed. These engineering
checks are not independent physiological validation.

```bash
gzip -dc artifacts/dna02-premotor-output-holdout43-45-e5a8d7d.json.gz | shasum -a 256
uv run python -m flybrain.premotor_output_probe run \
  artifacts/male-cns-v1.0-w5 \
  --odor-source-artifact artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz \
  --controls artifacts/dna02-premotor-controls-prior40-42-2841c9a.json \
  --output artifacts/reproduced-premotor-output.json
```

The narrow defensible conclusion is that the **collective, annotated
IN19A003/IN08A006 inhibitory output contributes to suppression of these
leg-rotator motor cells under the specified artificial ORN input in this
MaleCNS/Shiu model**. Six RNG seeds on one connectome and parameterization
are not an animal-population estimate. The required next bridge is an
independently calibrated motor-to-muscle and stride measurement with a
better activity-matched same-post control; the present result cannot be
promoted to living-fly steering or autonomous learning.
