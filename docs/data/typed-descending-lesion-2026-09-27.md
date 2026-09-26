# Typed DN lesion: primary motor criterion failed; direct targets were outside the decoder registry

This is a **negative prospective result** for the predeclared typed-DN motor
claim, plus a dataset-level anatomical finding that directs the next assay.
It is not evidence of learned navigation or real-fly physiology.

## Locked comparison

The [design](../superpowers/specs/2026-09-27-typed-descending-lesion-design.md)
was committed as `64d1e3c`, the [plan](../superpowers/plans/2026-09-27-typed-descending-lesion.md)
as `921dc99`, and the implementing code/tests as `57addff` **before** running
seeds 13–15. Each seed used four 100-body-step FlyGym 2.1.0 conditions:
DNg33 monitored but intact, DNg33 silenced, DNg48 monitored but intact, and
DNg48 silenced. All used the same MaleCNS snapshot, registries, arena,
Shiu/decoder parameters, and `--motor-trace`; phase-envelope amplitude was
zero. Only graph cells with `superclass=descending_neuron` and exact matching
`cell_type` could be silenced. Both types resolved to two acetylcholine cells.

The primary gate required, in **each** seed, positive baseline DNg33 spikes,
zero DNg33 spikes after lesion, exact replay and unchanged graph, at least 5%
fewer registered motor spikes after DNg33 lesion, and a DNg33 reduction at
least five percentage points larger than the DNg48 reduction. The rule was
not adjusted after viewing these runs.

| Seed | DNg33 spikes intact → lesion | Registered motor spikes intact → DNg33 lesion | DNg33 reduction | Motor spikes after DNg48 lesion | DNg48 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 602 → 0 | 2,716 → 2,707 | 0.33% | 2,716 | 0.00% |
| 14 | 601 → 0 | 2,644 → 2,541 | 3.90% | 2,644 | 0.00% |
| 15 | 598 → 0 | 2,653 → 2,717 | −2.41% | 2,653 | 0.00% |

The primary gate failed **0/3 seeds**: DNg33 never met the 5% motor-spike
reduction, and the effect reversed at seed 15. DNg48 produced zero spikes in
all three intact episodes, and silencing it left the complete motor/body trace
unchanged. Thus DNg48 was matched in cell count, transmitter, and nearly in
graph outdegree, but was **not an activity-matched functional control** in
this protocol. The two readout-only baselines had byte-identical canonical
motor-trace JSON within each seed; simply selecting a type did not change
the episode. All 12 runs replayed exactly, left the graph unchanged, and
carried clean software revision `57addff2823bb27488c20fba5b040f1fc26cb4dc`.

As secondary diagnostics, the time-integrated absolute applied joint moment
was 24.006 → 22.677, 22.741 → 22.521, and 23.785 → 22.179 µN·m·s for
intact → DNg33 lesion in seeds 13–15. These differences do not rescue the
failed primary gate. Every condition had zero food/threat contacts and DAN
events, unchanged KC→MBON multipliers, and at least one transient `fallen`
frame. No learning or stable biological gait was demonstrated.

## Why the registered-motor metric may miss this circuit

A direct retained-graph query found that the two DNg33 cells (13317, 13442)
each connect to the same eight `vnc_motor` cells: 800659, 803732, 810086,
813291, 814430, 814989, 815205, and 815281. These are annotated MNad03,
MNad22, or MNad25. There are 16 directed DNg33→motor edges with combined
signed graph weight +404. All eight targets are outside the current
36-population motor registry, which covers 182 of the 708 annotated
`vnc_motor` cells. DNg48 has two direct `vnc_motor` edges, one into that
registry, but its silence here prevents an activity-matched comparison.

The anatomical edge count is a **dataset measurement** and the lesion result
is a **simulation observation**. Direct synapses plus DNg33 spiking do not
show that any of these eight targets fired, that DNg33 caused their firing,
or that they drove muscles. The next gate is to count those target-cell
spikes under DNg33 lesion and matched non-target controls, with fixed
prospective seeds and no decoder remapping or gain change.

## Provenance and artifacts

All 12 records share snapshot SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
learning registry SHA-256
`7953875f6df1c655876090b3cd32ca71601e3463f8e5d7789e9f1c47e30c7750`,
motor registry SHA-256
`bd51ac1a9af8f582c66033a2cc7e15d595b799f22f500983ba7d0198f2b2de01`,
and interface registry SHA-256
`9b9ad410b0b86f1f272f94b8ef3d7e4351b8b3bd2f7e9836a52044a39d258e09`.
Outputs were losslessly compressed with `gzip -k -n -9`. SHA-256 below is of
the **uncompressed JSON**; check with `gzip -dc FILE.json.gz | shasum -a 256`.

| Seed | Condition | JSON.gz | JSON SHA-256 |
| ---: | --- | --- | --- |
| 13 | DNg33 lesion | [artifact](../../artifacts/typed-descending-flygym-seed13-dng33-lesion.json.gz) | `b5f25ccb622dccb218bc514c5c65d9784056dd165b64f66e47d2ceec58325adf` |
| 13 | DNg33 normal | [artifact](../../artifacts/typed-descending-flygym-seed13-dng33-normal.json.gz) | `8adf0985e8a1c7d8c80e5fb54dc02b019711100a12e005d4338e776be6d94048` |
| 13 | DNg48 lesion | [artifact](../../artifacts/typed-descending-flygym-seed13-dng48-lesion.json.gz) | `3366e54c4e7c353fd4f367a3b4bbcada7d50d21e49cf3a5fe08f8940197f04a9` |
| 13 | DNg48 normal | [artifact](../../artifacts/typed-descending-flygym-seed13-dng48-normal.json.gz) | `69b7dc710c47be2d30c46c4df9ccbf6265d065b765ad64c3631794f2842047c2` |
| 14 | DNg33 lesion | [artifact](../../artifacts/typed-descending-flygym-seed14-dng33-lesion.json.gz) | `5bb0233a5c81ba2f6d6ac6e6108e74ac6f848bb78fb4f75689beef2a209a0229` |
| 14 | DNg33 normal | [artifact](../../artifacts/typed-descending-flygym-seed14-dng33-normal.json.gz) | `98ab024d903b9a969b5fb3ede28e4f34e22c4ebdc5489ba6a6be67d5895960a8` |
| 14 | DNg48 lesion | [artifact](../../artifacts/typed-descending-flygym-seed14-dng48-lesion.json.gz) | `f80a6513754230510362322f15212e74323870a53c026f5282c4393708782470` |
| 14 | DNg48 normal | [artifact](../../artifacts/typed-descending-flygym-seed14-dng48-normal.json.gz) | `0032749b1a904b46d71e3242ffbdb8a9e71b0c638e67f269f09362d6f922f8c5` |
| 15 | DNg33 lesion | [artifact](../../artifacts/typed-descending-flygym-seed15-dng33-lesion.json.gz) | `eb067f3b1cfd723bd6236c0a4841f9706cf2d3c21c339da5ac379d491f49c76a` |
| 15 | DNg33 normal | [artifact](../../artifacts/typed-descending-flygym-seed15-dng33-normal.json.gz) | `ae7e074ad6d3a992d92d01733502c5e026419f52a6669d89588760ed76cc3ec0` |
| 15 | DNg48 lesion | [artifact](../../artifacts/typed-descending-flygym-seed15-dng48-lesion.json.gz) | `cff331a39f1b7dd3c4d2d9e957020663bc8332608f8a084e7e17b879067d08ba` |
| 15 | DNg48 normal | [artifact](../../artifacts/typed-descending-flygym-seed15-dng48-normal.json.gz) | `9d5e5eee5cf1c3ef640bbbe9d668c8c46274d66a3186a40aa754a81d29223b22` |

To reproduce one pair at code revision `57addff` after acquiring/importing
the four official MaleCNS products as in the [README](../../README.md):

```bash
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 13 --backend flygym --motor-trace \
  --descending-type DNg33 --descending-lesion none \
  --output artifacts/repro-typed-dng33-normal.json
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 13 --backend flygym --motor-trace \
  --descending-type DNg33 --descending-lesion annotated_type \
  --output artifacts/repro-typed-dng33-lesion.json
```
