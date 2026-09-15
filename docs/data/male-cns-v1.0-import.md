# MaleCNS v1.0 Canonical Import

## Verified run

Run on 2026-09-16 from the four SHA-256-verified official artifacts using:

```bash
/usr/bin/time -l uv run flybrain snapshot import-malecns \
  data/manifests/male-cns-v1.0-essential.json \
  --cache-root data/cache \
  --output artifacts/male-cns-v1.0-w5 \
  --min-weight 5
```

The selection rule matches the official `flyconnectome/2025malecns` quantification notebook:
non-empty superclass excluding values containing `tbc`, followed by directed connection
`weight >= 5` and membership of both endpoints in the selected population.

| Metric | Measured value |
|---|---:|
| source annotation rows | 211,577 |
| source weight rows processed | 151,856,684 |
| selected neurons | 166,606 |
| connected neurons after threshold | 165,768 |
| retained directed edges | 6,240,402 |
| retained total synaptic weight | 89,839,298 |
| untyped neurons | 2,103 |
| selected neurons missing transmitter rows | 166 |
| unresolved-sign edges | 1,177,887 |
| unresolved-sign synaptic weight | 16,259,452 |
| adapter runtime | 21.14 seconds |
| measured maximum RSS | 1,769,472,000 bytes |
| canonical snapshot size | 47 MiB |
| CSR arrays after load | 76,217,680 bytes |

The adapter's internal peak-RSS field reported 1,769,259,008 bytes; independent `/usr/bin/time -l`
reported 1,769,472,000 bytes. Both are below the 8 GiB acceptance ceiling. The filesystem retained
about 30 GiB free, above the mandatory 10 GiB reserve.

## Validation

`validate_malecns_snapshot()` independently scanned all retained Parquet edges and confirmed:

- exact canonical neuron and edge schemas;
- unique neuron IDs;
- every edge endpoint exists in the selected neuron table;
- every sign is one of `-1`, `0`, or `1`;
- retained edge, weight, unresolved-sign, and connected-neuron counts match `metadata.json`.

`SparseConnectome.from_snapshot()` loaded 166,606 neurons and 6,240,402 explicit CSR entries. A
zero-input propagation smoke test produced no activity, as required for deterministic dynamics.

## Scientific interpretation

The 166,606 selected neurons are a measured consequence of the v1.0 exports and published
selection mechanism, not a hard-coded target. The difference from counts quoted in papers or media
can reflect release, annotation, and filtering definitions.

Acetylcholine is represented as excitatory and GABA/histamine as inhibitory under an explicit
fast-transmitter assumption. Glutamate remains unresolved without receptor context. Dopamine,
octopamine, serotonin, and unclear annotations also remain unresolved because treating them as a
single instantaneous excitatory or inhibitory sign would be biologically misleading. These
zero-sign edges remain in canonical Parquet and are counted; receptor-aware dynamics and
neuromodulation are a subsequent phase.
