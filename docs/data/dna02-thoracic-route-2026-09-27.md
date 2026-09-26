# DNa02 reaches thoracic leg-motor cells, but neither new primary gate passes

This is a retained-MaleCNS **model-level** anatomical and intervention audit, not
an experiment on a living fly. It follows the [DNg33 annotation correction](dng33-abdominal-motor-identity-2026-09-27.md):
those direct targets are abdominal, so they cannot be treated as identified leg
motors. The DNa02 route below was selected from a scan of annotated descending
sources with direct positive edges into the existing 182-cell leg motor registry.
The exact thresholds, controls, and new seed ranges were stated in the task
transcript before the outcomes were read; unlike earlier DNg33 assays, this
one-off diagnostic was **not** source-committed as a separate protocol before
the outcome runs. The gates are reported unchanged despite failing.

## Anatomy and provenance

The two DNa02 sources are body IDs `10360` (right) and `523769` (left), both
`Prelim Roughly traced`. They have 17 positive direct edges to 17 distinct
registered `vnc_motor` cells, with project signed weight sum +874. All 17
targets have T1–T3 somata and `Reviewed`/`Traced` annotations. Twelve are
`Sternal anterior rotator MN`, two `Sternal posterior rotator MN`, two
`Ti extensor MN`, and one `Ti flexor MN`. The precise IDs, edge positions,
weights, and controls are frozen in the [isolated assay artifact](../../artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz).
The edge sign is a transmitter-based model assumption, not measured
postsynaptic physiology. The motor registry supplies an anatomical grouping;
it does not validate a neuron-to-muscle force law.

All new outputs report clean source revision
`644dfe6c45aedc4db337a26d01af54fa1f935672`, retained snapshot content
SHA-256 `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`,
166,606 neurons, and 6,240,402 directed model edges. The FlyGym runs have
identical learning, motor, and interface registry hashes across all nine
conditions. No decoder, graph weight, neural gain, or phase-drive setting was
changed to obtain these outcomes.

## Closed-loop FlyGym lesion: primary gate failed 0/3

Each seed 22–24 ran three 100-body-step, 0.01 s/step FlyGym 2.1.0 conditions:
intact DNa02, silenced DNa02 pair, and silenced active DNg13 pair. DNg13 is a
functional but **not activity- or anatomy-matched** control: it has only three
direct edges into the 182-cell registry, one sharing a DNa02 target. The
pre-outcome gate required source recruitment and silencing, at least 20% fewer
spikes in the 17 DNa02 targets, a reduction at least 10 percentage points
larger than both the other 691 `vnc_motor` cells and the DNg13-lesion target
reduction, exact replay, and unchanged graph in every seed.

| Seed | DNa02 spikes intact → lesion | 17 targets intact → DNa02 lesion → DNg13 lesion | DNa02 target reduction | Other 691 motor-cell reduction | DNg13 target reduction | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 22 | 272 → 0 | 689 → 550 → 577 | 20.17% | 0.75% | 16.26% | fail: DNg13 margin 3.92 pp |
| 23 | 229 → 0 | 689 → 571 → 640 | 17.13% | 4.84% | 7.11% | fail: below 20% |
| 24 | 254 → 0 | 700 → 576 → 561 | 17.71% | 2.67% | 19.86% | fail: below 20%; DNg13 stronger |

All nine runs replayed exactly, left the graph unchanged, and used zero
external phase-envelope amplitude. Every run had zero food/threat contacts,
zero contact-gated DAN events, unchanged KC→MBON multipliers, and at least one
transient fallen frame. Applied moment and final body coordinates changed, but
the DNa02 spikes also enter a manually specified, bounded descending-to-torque
decoder; body differences do **not** identify the 17 synapses as the movement
mechanism. The closed-loop lesion also changes sensory/proprioceptive feedback.
No learned appetitive/aversive behavior or stable biological gait is shown.

## Isolated direct-edge assay: direct pair effect, primary gate failed 0/3

The same two DNa02 source cells received seed-paired Poisson voltage input in
2,000 Shiu steps (200 ms) on the immutable MaleCNS graph. There was no body,
sensory feedback, plasticity, motor decoder, reward, planner, or RL. Conditions
were intact, zero multiplier on exactly the 17 direct DNa02→registered-motor
edges, zero multiplier on 17 edges from the **same sources** to distinct
`vnc_intrinsic` controls, no source input, and intact replay. The controls were
chosen before outcomes by minimum absolute synapse-count mismatch within each
source: total +887 versus +874 for targets (1.49% difference). The target IDs,
controls, CSR positions, and weights are in the artifact. The gate required
target recruitment, identical input events **and source-spike timing across all
three source-driven conditions**, at least 20% target reduction from direct-edge
zeroing, at least 10 percentage points more reduction than the matched-edge
control, exact replay, and unchanged graph in every seed.

| Seed | Intact DNa02 spikes | Target spikes intact → direct zero → control zero | Direct reduction | Control reduction | Why gate failed |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 55 | 71 → 38 → 86 | 46.48% | −21.13% | Control changed DNa02 spike timing/count: 55 → 56 |
| 26 | 59 | 74 → 39 → 105 | 47.30% | −41.89% | Control changed DNa02 spike timing/count: 59 → 61 |
| 27 | 66 | 81 → 54 → 117 | 33.33% | −44.44% | Control changed DNa02 spike timing/count: 66 → 64 |

The **intact versus direct-edge-zero pair** had identical external events and
identical DNa02 spike-time digests in all three seeds. Other 691 annotated
motor-cell aggregate spike counts were also unchanged in those pairs
(1,051/1,051; 1,504/1,504; 1,575/1,575); no-source target totals were zero.
These measurements support a narrower causal statement: under artificial
DNa02 input in this Shiu/MaleCNS model, those direct edges contribute to
recruitment of the anatomically thoracic target group. They do **not** show
that each target depends on its own edge, identify a muscle response, or show
natural stimulus-driven use of the route. Because removing the control edges
changed DNa02's own firing, the declared equal-source comparison with that
control is confounded. Its positive numerical contrast cannot rescue the
failed primary gate or be presented as controlled circuit specificity.

## Artifact integrity and reproduction

All ten compressed JSON files passed `gzip -t`; each digest below is SHA-256
of the **uncompressed** JSON. The nine FlyGym files were generated by the
existing `flybrain experiment autonomous-hexapod` command with `--steps 100`,
`--backend flygym`, `--motor-trace`, `--spike-readout-superclass vnc_motor`,
the stated seed, and `--descending-type DNa02`/`--descending-lesion none` or
`annotated_type`; DNg13 controls used type `DNg13` and lesion `annotated_type`.
The isolated run reused the tested `_condition` and sparse-overlay functions
from `flybrain.dng33_edge_probe`, with the frozen edges and Shiu parameters
recorded in its artifact (`dt_ms=0.1`, `refractory_ms=2.0`,
`synaptic_delay_ms=1.0`, other Shiu defaults). This one-off assay has no
dedicated committed CLI or tests, so it is weaker as an independently
reproducible publication than the DNg33 assay.

| Seed | Condition | Artifact | Uncompressed SHA-256 |
| ---: | --- | --- | --- |
| 22 | normal | [JSON.gz](../../artifacts/dna02-thoracic-seed22-normal.json.gz) | `172fb67dcfa90e697abaa490b0135b859b16d2d267d7b32e6c7f2465c6de9cef` |
| 22 | DNa02 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed22-dna02-lesion.json.gz) | `58a200862d0b706f662e493bf2b273f59c4683a15ba79a294f5f4b6966f4bd92` |
| 22 | DNg13 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed22-dng13-lesion.json.gz) | `e0a3959009c19d8dc057a608ebe5c752d991707249c34bbf3fb8db329ea60b03` |
| 23 | normal | [JSON.gz](../../artifacts/dna02-thoracic-seed23-normal.json.gz) | `fc90112a1cfb62ef3b948cd18a224c68d1934d0b4cf4a38abda6a56bd7b14f75` |
| 23 | DNa02 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed23-dna02-lesion.json.gz) | `767df6e08eadd925849d6a7ae63578f26abb91511a3bcc926d821e27b0b09d9f` |
| 23 | DNg13 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed23-dng13-lesion.json.gz) | `f4f2b24216dfc89452253d5d01249f24b1edfbf737792b3eddb38745263b82c6` |
| 24 | normal | [JSON.gz](../../artifacts/dna02-thoracic-seed24-normal.json.gz) | `dd047411580add729001e3460a8b3ef96bc0d8b3d4c51392ea1df406a08386a3` |
| 24 | DNa02 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed24-dna02-lesion.json.gz) | `327a345146471f09a3f7b5c595511de01338c6b1a4c156a599327f708b4eb8d4` |
| 24 | DNg13 lesion | [JSON.gz](../../artifacts/dna02-thoracic-seed24-dng13-lesion.json.gz) | `6dc667669dd82d356231b1e7962cc28e905da0d8f5c144a8b2f94471f47ee743` |
| 25–27 | isolated edges | [JSON.gz](../../artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz) | `3d2dc99233197e45d2849aa6b52220689e2b230fcb53f85bce991ac1ba8b0dc5` |

Check a row with `gzip -dc PATH.json.gz | shasum -a 256`. A fresh full local
test suite returned **488 passed, 28 skipped**; it tests the repository code,
not biological validity or this one-off protocol's independent reproducibility.

## Remaining claim boundary

This is a stronger anatomical match to the leg motor registry than the DNg33
abdominal route, and a replicated direct-edge **model** effect under fixed
artificial source input. It is not a passed specificity gate, a complete
muscle-actuation map, learned locomotion, food/threat learning, transfer to
games, or a complete autonomous fly. The existing
[autonomy evidence audit](autonomy-evidence-audit-2026-09-23.md) lists the
additional physiological calibration, unassisted stable motor control,
independent behavioral holdouts, causal learning controls, multi-sense and
internal-state validation still required. The latest retained 100-step
[reachability probe](autonomy-reachability-probe-2026-09-26.md) still reports
`behavioral_claim_allowed=false`; it had only one independent seed, no normal
food holdout contact, and mixed/negative causal-control effects.
