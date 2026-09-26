# DNg33's eight direct motor targets are AbN-labelled abdominal cells, not identified leg motors

This is a **post-hoc anatomical reinterpretation** of the [prospective
closed-loop target assay](dng33-direct-motor-targets-2026-09-27.md) and the
[predeclared fixed-source edge assay](dng33-direct-edge-localization-2026-09-27.md).
It does not add a new behavioral experiment. Both assays remain valid at
their stated model-level endpoints, but their direct target readout must not
be described as a leg-motor or muscle-output effect.

## Retained MaleCNS annotation audit

An independent join of `edges.parquet` and `source-annotations.parquet` in
snapshot SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`
found **all** direct `vnc_motor` edges from the two DNg33 cells (13317,
13442): 16 positive edges to exactly the eight targets below, combined
project signed weight +404. Both DNg33 neurons have the status label
`Prelim Roughly traced`; each listed target has `statusLabel=Reviewed` and
`status=Traced`. The edge sign uses the project's transmitter-sign assumption,
not measured postsynaptic physiology.

| Target body ID | Annotated type | Soma side | Soma neuromere | Exit nerve | Two DNg33 edge weights, L/R | Sum |
| ---: | --- | :---: | :---: | :---: | ---: | ---: |
| 800659 | MNad22 | R | A7 | AbNT | 14 / 7 | 21 |
| 803732 | MNad03 | R | A4 | AbN4 | 34 / 34 | 68 |
| 810086 | MNad03 | L | A4 | AbN4 | 16 / 21 | 37 |
| 813291 | MNad25 | R | A2 | AbN3 | 15 / 9 | 24 |
| 814430 | MNad03 | L | A5 | AbN4 | 42 / 39 | 81 |
| 814989 | MNad03 | R | A3 | AbN4 | 25 / 13 | 38 |
| 815205 | MNad03 | L | A3 | AbN4 | 29 / 29 | 58 |
| 815281 | MNad03 | R | A5 | AbN4 | 49 / 28 | 77 |

The snapshot contains 708 annotated `vnc_motor` cells, 238 with an `AbN`-
prefixed `exitNerve`. **All eight** direct DNg33 motor targets are in that
AbN-labelled subset, with somata in A2–A5 or A7; no direct DNg33→`vnc_motor` edge
in this retained graph targets an annotated thoracic leg motor cell. The
238/708 comparison is descriptive, **not** a predeclared enrichment test:
these eight targets and the connectome edges are not random independent
samples. Other DNg33 outputs could still affect leg circuits indirectly.
The annotations give no muscle identity, innervation strength, or force law
for these eight cells. Their `MNad` labels and `AbN` exit nerves are not a
license to route their spikes into the existing left-front-leg FlyMimic map.
None of the eight IDs occurs in the 182-cell v2 leg motor registry.

## Existing embodied activity, regrouped by annotated segment

The following sums regroup **only those eight target cells** in the
previously published seed-16–18 FlyGym JSON. They were not predeclared as a
separate primary endpoint. Every compressed input passed `gzip -t`, and its
uncompressed SHA-256 matched the earlier report. Values are intact →
DNg33-pair lesion spike totals:

| Target soma neuromere | Seed 16 | Seed 17 | Seed 18 | Pooled descriptive reduction |
| :---: | ---: | ---: | ---: | ---: |
| A2 (one cell) | 85 → 66 | 84 → 65 | 83 → 67 | 21.4% |
| A3 (two cells) | 141 → 31 | 139 → 26 | 137 → 32 | 78.7% |
| A4 (two cells) | 125 → 20 | 128 → 14 | 122 → 19 | 85.9% |
| A5 (two cells) | 199 → 72 | 199 → 61 | 197 → 72 | 65.5% |
| A7 (one cell) | 0 → 0 | 1 → 0 | 1 → 0 | too few spikes |

Thus the six A3–A5-labelled MNad03 cells account for 1,040 of the 1,096
fewer target spikes pooled across these three seeds (94.9%, descriptive and
post-hoc); A7 has insufficient embodied activity for an effect estimate.
Body feedback and polysynaptic paths were not isolated
in those runs. The separate fixed-source edge experiment established
necessity of the 16 direct connections for all eight target responses under
artificial DNg33 drive, not a segment-specific natural behavior.

## Consequence for the next biological gate

Do **not** connect these eight abdominal-labelled IDs to the existing
36-population leg-torque decoder or the tethered left-front-leg FlyMimic
muscles merely to obtain movement. A defensible next physical gate needs an
independently sourced abdominal muscle/nerve mapping and an actuator/body
model for that anatomy, or a different, independently identified thoracic
motor route. Until then, DNg33→these motor-cell spikes are a circuit result,
not a measured leg gait, muscle force, or autonomous behavior result.

To recheck the identity and base-rate counts on the retained snapshot:

```bash
.venv/bin/python - <<'PY'
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

path = 'artifacts/male-cns-v1.0-w5/source-annotations.parquet'
rows = pq.read_table(path, columns=[
    'bodyId', 'instance', 'type', 'somaSide', 'somaNeuromere',
    'exitNerve', 'superclass', 'statusLabel', 'status',
])
motor = rows.filter(pc.equal(rows['superclass'], 'vnc_motor')).to_pylist()
ids = [800659, 803732, 810086, 813291, 814430, 814989, 815205, 815281]
targets = rows.filter(pc.is_in(rows['bodyId'], value_set=pa.array(ids))).to_pylist()
print('motor', len(motor), 'AbN', sum(str(r['exitNerve']).startswith('AbN') for r in motor))
for row in sorted(targets, key=lambda r: r['bodyId']):
    print(row['bodyId'], row['type'], row['somaSide'],
          row['somaNeuromere'], row['exitNerve'], row['statusLabel'])
edges = pq.read_table('artifacts/male-cns-v1.0-w5/edges.parquet',
                      columns=['pre_id', 'post_id', 'synapse_count', 'sign'])
direct = edges.filter(pc.and_(
    pc.is_in(edges['pre_id'], value_set=pa.array([13317, 13442])),
    pc.is_in(edges['post_id'], value_set=pa.array(ids)),
)).to_pylist()
print('direct_edges', len(direct), 'signed_weight',
      sum(r['synapse_count'] * r['sign'] for r in direct))
PY
```
