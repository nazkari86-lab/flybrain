# Registered ORN stimulation recruits DNa02 and its direct thoracic motor targets in the MaleCNS model

This is a **narrow, positive model-circuit replication**, not an experiment on
living flies or proof of autonomous intelligence. The [protocol](../superpowers/specs/2026-09-30-dna02-registered-orn-edge-design.md)
was published as `67720fa87c258ffaf7a790d6cc59f5999b3bb1f7`, and the
[tested runner](../../src/flybrain/dna02_odor_probe.py) was published as
`a1ed69b6e68ab76bcc387e1ad50d6913f01c50ec`, **before** results for
previously unused seeds 37–39 were read. The 5% threshold was chosen after
exploratory seeds 34–36, so this is a transparent held-out replication of a
model effect, not independent hypothesis discovery.

## Observed route and fixed intervention

The original seed-7-to-9 registered-ORN panel was rerun without graph or
parameter changes. All six food/threat full-neural-trace SHA-256 digests
matched the [published ORN artifact](../../artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz).
The two DNa02 cells (`10360`, `523769`) fired 22/27/29 times under the
food-labeled ORN input and 19/19/19 times under the threat-labeled ORN input;
the 17 fixed thoracic motor targets fired 121/136/128 and 100/95/110 times,
respectively. This establishes recruitment in this existing neural model,
not real odor perception. The food/threat labels are task assignments to
registered ORN_DM1/VA2 and ORN_DA2 populations.

In the new-seed panel, the exact same Poisson ORN events were reused across
intact, 17 direct-edge-zero, and 17 same-source-control-edge-zero conditions.
No afferents to DNa02 were removed; DNa02 itself received **no external
voltage events**. Each condition ran 2,000 Shiu steps (200 ms). The direct
intervention zeroed only the [previously frozen](../../artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz)
17 positive DNa02-to-`vnc_motor` edges. No-source and intact-replay conditions
were also run for every seed/odor pair. The retained graph has 166,606 cells
and 6,240,402 edges and was unchanged.

| ORN input | Seed | DNa02 spikes | 17 target spikes: intact → direct zero → matched zero | Direct reduction | Matched control kept DNa02 timing? | Narrow gate |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| Food-labeled | 37 | 22 | 132 → 110 → 117 | 16.67% | no | pass |
| Food-labeled | 38 | 24 | 131 → 115 → 125 | 12.21% | no | pass |
| Food-labeled | 39 | 30 | 126 → 104 → 137 | 17.46% | no | pass |
| Threat-labeled | 37 | 19 | 90 → 81 → 108 | 10.00% | no | pass |
| Threat-labeled | 38 | 28 | 120 → 108 → 112 | 10.00% | no | pass |
| Threat-labeled | 39 | 27 | 99 → 83 → 118 | 16.16% | no | pass |

The predeclared **narrow** gate passed **6/6**: ORN-event digest and exact
observed DNa02 spike-time digest matched in each intact/direct pair; the
other `vnc_motor` cells' aggregate spike count was unchanged; direct-edge
zeroing reduced target spikes by at least 5%; no-source yielded zero DNa02
and target spikes; intact replay reproduced every recorded neural readout;
and the graph digest did not change. A separate read-only calculation from
the complete JSON, without trusting its `narrow_gate_passed` fields,
recomputed all six results.

**The stronger matched-control comparison remains invalid.** Zeroing the
same-source `vnc_intrinsic` controls changed DNa02's own spike-time digest
in **6/6** new-seed conditions, just as it did in exploratory food seeds
34–36. The earlier strict matched-control gate failed 0/3. The positive
numerical direct-versus-control contrasts above cannot be interpreted as
controlled pathway specificity; the control intervention perturbs the
presynaptic source. The [previous afferent-isolated assay](dna02-afferent-isolated-2026-09-30.md)
passed a different 6/6 matched-control gate only by zeroing all 1,136
afferents into DNa02 and artificially stimulating it. The present result is
stronger on *registered upstream input*, weaker on *matched-control
specificity*. Neither proves behavior.

## Reproduction and limits

The [full compressed JSON](../../artifacts/dna02-registered-orn-seeds37-39.json.gz)
contains all five conditions, edge identities/positions, per-target counts,
event and neural digests, gates, graph and input hashes, software revision,
panel runtime excluding graph load (49.2 s), and peak RSS (1.02 GB).
`gzip -t` passed; the SHA-256 of
the **uncompressed** JSON is
`6e8a30b60169fff689635af5ff1237ed99bd99c80886e5fa659f3759cfd1f88d`.
Check the bytes with:

```bash
gzip -dc artifacts/dna02-registered-orn-seeds37-39.json.gz | shasum -a 256
```

To regenerate a new local JSON without overwriting the published artifact:

```bash
.venv/bin/python -m flybrain.dna02_odor_probe \
  artifacts/male-cns-v1.0-w5 \
  --frozen-edge-artifact artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz \
  --odor-source-artifact artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz \
  --output artifacts/reproduced-dna02-registered-orn.json \
  --steps 2000 --seeds 37 38 39
```

The code revision passed the full local suite (`494 passed, 28 skipped`),
Ruff, mypy, and a two-file CodeRabbit review with zero findings before the
held-out run. These are software checks, **not** biological validation.
The defensible statement is that, under artificial registered ORN voltage
input and intact DNa02 afferents, these 17 direct edges contribute causally
to thoracic motor-cell spike recruitment in **one** Shiu/MaleCNS model and
parameterization. This does not establish natural odor responses, a unique
MBON route, muscle force, leg motion, learning, stable gait, or autonomous
intelligence. The earlier closed-loop DNa02 primary gate failed 0/3 and
remains negative evidence. Model signs, membrane dynamics, receptor input,
and source annotations retain their stated uncertainties; six simulation
seeds are not six animals or a population confidence interval.
