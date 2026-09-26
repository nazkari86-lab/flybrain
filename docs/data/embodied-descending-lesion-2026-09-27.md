# Annotated descending neurons causally affect the embodied MaleCNS simulation

This is a reproducible, model-level **biological-circuit intervention**, not
evidence of learned navigation or of an intelligent real fly. The retained
MaleCNS snapshot has 166,606 neurons and 6,240,402 directed edges. FlyGym
2.1.0 supplies the body, but the Shiu membrane model, 36-population neural
motor decoder, joint torque authority, odor field, and partial 18/66 leg-DOF
bridge are engineering assumptions. No hidden RL policy or gait controller
was used. The phase envelope amplitude was zero.

The previous [isolated ORN result](natural-odor-descending-lesion-2026-09-27.md)
showed that silencing all annotated DNs abolished registered motor spikes under
fixed receptor-population stimulation. The present **closed-loop** experiment
is different: the body moves, vision and proprioception are active, and the
odor interface uses the broad `bipartite_registered_olfactory_assumption`, not
the receptor-specific ORN_DM1/VA2/DA2 source panel. Because body states diverge
after a lesion, sensory input events need not remain identical between paired
conditions. Do not concatenate the two assays into a proven specific
ORN→MBON→DN→muscle chain.

## Pre-specified replication and procedure

Exploratory seeds 7–9 used source revision `71679d7+dirty` while the lesion
code was being finalized. The design and prospective predictions were committed
as [`dd3fd1e`](https://github.com/nazkari86-lab/flybrain/commit/dd3fd1e22d1569fb5b5add8b7310d0b29aa56051)
**before** running seeds 10–12. The committed
[design](../superpowers/specs/2026-09-27-descending-lesion-body-design.md)
required, separately in all three new seeds, zero annotated-DN spikes, at
least 50% fewer registered motor spikes, and at least 40% less time-integrated
absolute applied joint moment in the all-DN lesion than the paired baseline.
The thresholds were chosen from exploratory results; the new seeds are a
prospective replication of that effect, not independent hypothesis generation.
Three simulation seeds do not provide an animal-level effect estimate or a
population confidence interval. An equally sized, activity-matched non-DN
lesion was not run, so DN specificity relative to such a control is unproven.

Each run used the same MaleCNS snapshot, registry versions, seed, arena, FlyGym
body, 100 body steps at 0.01 s/step, and `--motor-trace`. The experimental
intervention silenced exactly the 1,314 graph neurons annotated with
`superclass=descending_neuron` inside the existing Shiu mask. It did not edit
edges, plastic weights, decoder gains, or target coordinates. The seed-10
passive control silenced all 36 registered motor populations, yielding zero
applied torque. Every condition performed exact replay and left the canonical
graph unchanged.

Clean-source confirmatory artifacts record software revision
`dd3fd1e22d1569fb5b5add8b7310d0b29aa56051`, snapshot SHA-256
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
learning registry SHA-256
`7953875f6df1c655876090b3cd32ca71601e3463f8e5d7789e9f1c47e30c7750`,
motor registry SHA-256
`bd51ac1a9af8f582c66033a2cc7e15d595b799f22f500983ba7d0198f2b2de01`,
and interface registry SHA-256
`9b9ad410b0b86f1f272f94b8ef3d7e4351b8b3bd2f7e9836a52044a39d258e09`.

## Measurements

`Motor spikes` below is the sum of registered motor-neuron spikes over 100 body
steps. `Moment integral` is `0.01 s × Σ(body step, 18 mapped joints) |applied
torque|` in µN·m·s. The 18-joint adapter does not measure real muscle force.

| Seed | Motor spikes baseline → all-DN lesion | Reduction | Moment integral baseline → lesion (µN·m·s) | Reduction | Annotated DN spikes baseline → lesion |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2,727 → 1,029 | 62.27% | 22.276 → 9.366 | 57.96% | 30,258 → 0 |
| 11 | 2,722 → 1,068 | 60.76% | 24.617 → 10.480 | 57.43% | 30,398 → 0 |
| 12 | 2,731 → 1,071 | 60.78% | 22.933 → 10.910 | 52.43% | 30,007 → 0 |

The locked model-level prediction passed **3/3 prospective seed pairs**. The
lesion also reduced active registered motor groups, but did not silence all
motor cells. This contrasts with the earlier fixed-source isolated assay:
registered motor spikes remain under all-DN silencing, so other motor drive is
present in this embodied model. The route carrying it is unidentified. This is
a broad necessity test for part of the motor drive,
not localization of a minimal DN, MBON, or sensory route.

Seed 10 checks physical output against passive settling. From body step 20
to 100, the thorax's X displacement was +0.1197 mm (baseline), +0.1117 mm
(all-DN lesion), and +0.00185 mm (all 36 motor populations silenced). Thus
both neural conditions produced motion beyond this passive comparator; the
two active trajectories also ended at different thorax positions. The all-DN
lesion did not improve *desirable* behavior: across seeds 10–12, neither
condition contacted food or threat, no contact-gated DAN events occurred, and
the KC→MBON multipliers remained at unity. In fact the lesioned body ended
closer to food than the intact one in all three seeds, which cannot be
interpreted as learned food seeking. Every run had at least one transient
`fallen` frame; some intact bodies ended with only 3–5 supporting legs. End
stability and biological gait are not established.

## Reproduction and artifacts

Use source revision `dd3fd1e` with the retained snapshot and registry files.
The four official input products, their checksums, and the import commands are
in [the source manifest](../../data/manifests/male-cns-v1.0-essential.json)
and [the import instructions](../../README.md#import-the-real-malecns-v10-graph).
For a paired seed, run:

```bash
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 10 --backend flygym --motor-trace \
  --descending-lesion none --output artifacts/repro-dn-normal-seed10.json
.venv/bin/flybrain experiment autonomous-hexapod artifacts/male-cns-v1.0-w5 \
  --steps 100 --seed 10 --backend flygym --motor-trace \
  --descending-lesion all_annotated --output artifacts/repro-dn-lesion-seed10.json
```

For the passive comparator, repeat the baseline command with all 36
`--motor-lesion-group` names from
`data/registry/hexapod-motor-registry-v2.json`; its artifact records the exact
list. Outputs are gzip-compressed losslessly (`gzip -n -9`). The SHA-256 values
below refer to **uncompressed JSON bytes**; verify with
`gzip -dc <artifact.json.gz> | shasum -a 256`.
For each artifact, `jq '.episode | {motor_spikes,
annotated_descending_spikes, silenced_descending_neurons, replay_exact,
graph_unchanged}'` on the decompressed JSON prints the core causal readouts.

| Condition | Artifact | SHA-256 of JSON |
| --- | --- | --- |
| Seed 10 normal | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed10-normal.json.gz) | `c5a760812a4ea5250cc25699ced9a18b19ecccdc3600e63192ed037a3a400880` |
| Seed 10 all DN | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed10-all-dn.json.gz) | `2d0c118cf386d35a8a29626323b23eac30432e2270d39886e708dfbbe9fa66a8` |
| Seed 10 all motor | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed10-all-motor.json.gz) | `e763855a2388cdd6a82d4583bcf01bf8e18e96f251b0371dbe9f0562a3157f72` |
| Seed 11 normal | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed11-normal.json.gz) | `c0a39d67bc6c79a0236d44f86c12d60ddfb23b8fdcbef61ae99b3f08c4ae5f75` |
| Seed 11 all DN | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed11-all-dn.json.gz) | `92395b6548964f24caf8dfc3a5f3b1131b962041ddafe2bbc54da01478eb7dad` |
| Seed 12 normal | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed12-normal.json.gz) | `600eaef239f88f863b5e1203d3ef382487a299bc209341427ce21f30496c1605` |
| Seed 12 all DN | [JSON.gz](../../artifacts/descending-lesion-flygym-confirm-seed12-all-dn.json.gz) | `6f5d818cc1a45039280ef8d2293344a8e43296f7c0e835eabfd622a9cf66c2dd` |

The earlier exploratory seeds 7–9 are retained as
`artifacts/descending-lesion-flygym-seed{7,8,9}-steps100-{normal,all-dn}.json.gz`,
with a seed-7 all-motor control. They gave the same qualitative motor reduction
but had dirty-revision provenance and were *not* counted toward the prospective
3/3 result.

The next biological gate is specificity: matched lesions of candidate DN
subpopulations and homologous controls, followed by genuinely learned food and
threat behavior against no-plasticity, DAN, KC→MBON, and rewired controls on
independent arenas/seeds. This report does not satisfy that gate.
