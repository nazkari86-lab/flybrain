# Afferent-isolated DNa02 edges selectively recruit thoracic leg-motor targets in the MaleCNS model

This is a **positive, preregistered model-circuit result**, not a living-fly
experiment or a claim of walking, learning, or autonomous intelligence. The
[protocol](../superpowers/specs/2026-09-30-dna02-afferent-isolated-edge-design.md)
was committed and published as `e0f2457a24d2ff51fdeebd1c71a3c482e8e90a91`
before outcome seeds 28–30. It addresses the exact failure of the
[previous DNa02 assay](dna02-thoracic-route-2026-09-27.md): removing its
weight-matched control edges changed DNa02's own spike train.

## Frozen intervention

The two DNa02 cells are `10360` and `523769`. The 17 frozen positive edges
from these sources to 17 distinct, registered T1–T3 `vnc_motor` targets have
combined model weight +874. The control comprises 17 positive edges from the
**same sources** to `vnc_intrinsic` cells, weight +887. The exact source/post
IDs, weights, and CSR positions came unchanged from the already published
[frozen edge artifact](../../artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz),
uncompressed SHA-256
`3d2dc99233197e45d2849aa6b52220689e2b230fcb53f85bce991ac1ba8b0dc5`.

All five conditions zeroed the same 1,136 canonical edges **into** the two
DNa02 cells (579 into `10360`, 557 into `523769`) using a sparse overlay.
This kept recurrent network feedback from altering the source spike train.
The intact condition had only that common overlay; the direct condition also
zeroed the 17 target edges; the matched condition instead zeroed the 17
control edges. No-source omitted external DNa02 voltage events; replay
repeated intact. The source Poisson events were generated once per seed and
reused. Each run used 2,000 Shiu steps (200 ms), `dt_ms=0.1`,
`refractory_ms=2.0`, `synaptic_delay_ms=1.0`, and otherwise unchanged Shiu
defaults. There was no body, sensory feedback, plasticity, motor decoder,
reward, planner, or RL. The canonical graph was not modified.

## Predeclared primary result

Every seed had to recruit DNa02 and target cells above no-source, preserve
both the external-input digest and the **observed DNa02 spike-time digest**
across intact, direct-edge-zero, and matched-edge-zero conditions, reduce
target spikes by at least 20% after direct-edge zeroing, exceed the matched
control reduction by at least 10 percentage points, replay exactly, and leave
the canonical graph unchanged. All gates passed in **3/3 seeds**.

| Seed | DNa02 spikes, identical in paired conditions | 17 targets: intact → direct zero → matched zero | Direct reduction | Matched reduction | Other 691 motor-cell spikes: intact → direct zero | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 28 | 42 | 84 → 59 → 101 | 29.76% | −20.24% | 1,468 → 1,468 | pass |
| 29 | 50 | 69 → 49 → 89 | 28.99% | −28.99% | 1,435 → 1,435 | pass |
| 30 | 40 | 53 → 38 → 82 | 28.30% | −54.72% | 1,256 → 1,256 | pass |

No-source target totals were zero. Intact replay matched its full recorded
readout and neural trace in every seed. The source-event and source-spike-time
digests were equal across the three source-driven conditions, not merely the
total source spike counts. A separate read-only recomputation from the
artifact reproduced every gate and independently reconstructed the 1,136
afferent CSR positions from the retained snapshot.

## Exact claim boundary

Under **artificial DNa02 drive with all DNa02 afferents removed**, these 17
anatomically thoracic direct edges make a selective causal contribution to
target-cell firing in this Shiu/MaleCNS model. The matched control shares
presynaptic identity, edge count, and approximate weight, but not downstream
connectivity or activity; this does not prove unique physiological function
for each edge. All 1,136 source afferents were disabled, so this is **not**
an intact circuit or natural sensory response. Both DNa02 sources have only
`Prelim Roughly traced` source annotations, though the 17 motor targets are
`Reviewed`/`Traced`. Transmitter-based edge signs and homogeneous Shiu
dynamics are model assumptions. Three seeds on one connectome and one
parameterization are not a population confidence interval. No muscle force,
leg trajectory, learned behavior, or animal-equivalent intelligence was
tested. The project's autonomous-behavior claim remains false.

## Artifact and integrity

The [compressed full JSON](../../artifacts/dna02-afferent-isolated-seeds28-30.json.gz)
passed `gzip -t`; its uncompressed SHA-256 is
`6b2deb8ca0ea60fdba55d8cd34f0defee794c7d4d0efe8331b0541cdbdff8e04`.
It records every edge, overlay position, condition, per-cell target count,
source/event/trace digest, gate, source revision
`e0f2457a24d2ff51fdeebd1c71a3c482e8e90a91`, and snapshot content
SHA-256 `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`
(166,606 neurons; 6,240,402 model edges). Verify the bytes with:

```bash
gzip -dc artifacts/dna02-afferent-isolated-seeds28-30.json.gz | shasum -a 256
```

The seed-28–30 artifact was run through existing sparse-overlay and Shiu
helpers using a one-off script. The later
`src/flybrain/dna02_afferent_probe.py` runner was **not** used to generate
that artifact. Before the new-seed source lock, the runner was executed on
seeds 28–30 and its full per-seed condition/gate data, overlay positions,
edge lists, and graph digest matched the original JSON exactly; only the
software-revision field differed, and the runner additionally records
`control_ids`. This is a code-path reproducibility check, not an independent
biological replicate. Unused-seed replication remains a separate gate.
Neither check expands the biological interpretation beyond the
afferent-isolated model circuit.

The later runner invocation is:

```bash
.venv/bin/python -m flybrain.dna02_afferent_probe \
  artifacts/male-cns-v1.0-w5 \
  --frozen-edge-artifact artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz \
  --output artifacts/reproduced-dna02-afferent.json \
  --steps 2000 --seeds 28 29 30
```
