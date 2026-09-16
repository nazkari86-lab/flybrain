# MaleCNS biological steering assay

## Scope

This assay replaces arbitrary embodied neuron IDs with exact populations resolved from the retained
MaleCNS v1.0 annotation table. It separately tests direct descending-neuron calibration,
photoreceptor input, explicit HS optic-flow and LC16 looming feature calibrations, causal lesions,
restoration, replay, perturbation, holdout geometry, and a reduced feature-bypass closed loop.

The HS and LC16 protocols directly stimulate feature cells and therefore bypass unmodelled retina
and optic-lobe processing. They are not retina-to-behavior evidence. The legacy `embodied-loop`
command is an `arbitrary-smoke-only` plumbing test and permits no behavior claim.

## Exact retained run

```bash
PYTHONPATH=src .venv/bin/flybrain experiment biological-steering \
  /Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
  --registry data/registry/biological-interface-registry-v1.json \
  --steps 40 \
  --seed 7 \
  --output /Users/dulatnurlanuly/Downloads/flybrain/artifacts/biological-steering-malecns-seed7.json
```

The opt-in real-data gate was also run with:

```bash
FLYBRAIN_MALECNS_SNAPSHOT=/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5 \
  PYTHONPATH=src .venv/bin/pytest tests/test_biological_steering_real.py -q
```

It completed with `1 passed in 3.56s`.

## Immutable identities

| Field | Value |
| --- | --- |
| Software revision | `633d16dfcf3b6fb83e3bdcd4f17caa07d3474efc` |
| Dataset | `male-cns-v1.0-essential` |
| Registry SHA-256 | `969912a9ac9a69438aa880198c119e463bb984763a8aabd58a6534d7708db41a` |
| Source manifest SHA-256 | `8406eacdb75db4f1b84cfbf13b281b51628e68941458504312c0f42329fd5122` |
| Snapshot content SHA-256 | `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f` |
| Annotation SHA-256 | `ead87b4d37f97968f2845d525309649a1db9c7e8a032b147af32d51318eaa67b` |
| Result SHA-256 | `5c0e8aec0b73e3e24a532b30f25e23d097e3472bc7aa11306ef40d83f2827344` |
| Neurons | 166,606 |
| Directed edges | 6,240,402 |
| Conditions | 27 |

The retained result is generated under `artifacts/` and intentionally remains untracked because
that directory is reserved for reproducible local outputs.

## Measured outcomes

All nine direct-calibration gates passed: left/right ipsiversive signs, bilateral cancellation,
matching steering silence, exact restoration, bilateral MDN retreat, MDN silence, exact replay,
and graph immutability. Direct DNa02-left turn integral was `1.0`, its lesion value was `0.0`, and
bilateral MDN reverse integral was `2.0`, reduced to `0.0` by the matching lesion.

| Claim key | Classification | Measured result | Boundary |
| --- | --- | --- | --- |
| `photoreceptor_response` | `null` | No declared DN spikes or turn from hemispheric R1-R6 luminance/contrast | Upstream retina was not bypassed, but there is no ommatidial map |
| `hs_optic_flow` | `positive` | Left/right turns were `+0.5/-0.5`; matching lesion reduced `+0.5` to `0.0`; restoration, replay, and two holdouts agreed | Direct HS feature calibration bypassed upstream vision |
| `lc16_looming` | `null` | No MDN spikes and no reverse integral | Direct LC16 feature calibration bypassed upstream vision |
| `feature_closed_loop` | `positive` | Turn integral `0.5`, matching DNa02-left lesion `0.0`, exact replay | Reduced closed loop still uses the explicit HS/LC16 bypass |

The real nulls were retained unchanged. The HS positive result is evidence for the simulated
HS-to-descending feature path under this model, not evidence that the model sees optic flow through
its photoreceptors. The R1-R6 null means this phase does not establish retina-to-behavior steering.

## Resource and integrity audit

The retained CLI run reported `5.608 s` runtime and `793,362,432` bytes peak RSS. A read-only audit
re-resolved all 16 populations, compared the complete registry manifest embedded in the output,
checked all required lesion/restoration/replay/holdout condition names, required finite metrics and
positive denominators, and recomputed the result SHA-256. A source scan found no dense
`neuron_count x neuron_count` allocation in the runtime or tests; the canonical graph remains CSR.

## Scientific limitations

- Photoreceptor stimulation is hemispheric, not ommatidial or retinotopic.
- Each bank receives a fixed total voltage divided across its neurons, so population size does not
  multiply total injected drive; this explicit model assumption can contribute to the R1-R6 null.
- HS optic-flow and LC16 looming are explicit feature calibrations that bypass upstream visual
  processing.
- The body is a reduced planar model, not six-leg biomechanics or muscle control.
- Forward walking uses a tonic, target-independent model assumption.
- The fixed decoder is evidence-bound but is not a learned policy.
- This assay does not establish food seeking, threat avoidance, general intelligence, consciousness,
  or animal-equivalent behavior.
- Mushroom-body plasticity exists in a separate validated experiment; this assay does not show that
  learning controls the embodied steering loop.

The next biological gates are a phase-aware motor-neuron/muscle interface, richer compound-eye
geometry, and learned food/threat behavior with predeclared lesions and holdout worlds.

## Requirement coverage audit

| Design requirement | Implementation | Test evidence | Retained evidence |
| --- | --- | --- | --- |
| Typed evidence and exact population resolution | `biological_registry.py` | `test_biological_registry.py` | Embedded 16-population resolved registry and hashes |
| Label-free retinal observation and feature boundary | `retinal_interface.py` | `test_retinal_interface.py` | Separate photoreceptor and feature protocols with bypass flags |
| DN-only bounded decoder and named silence | `descending_interface.py` | `test_descending_interface.py` | DNa02/DNg13/MDN calibration and lesion conditions |
| Direct calibration separated from sensory claims | `steering_benchmark.py` protocol enum | `test_steering_benchmark.py` zero-edge calibration fixture | `dn_calibration` conditions and four separately classified sensory keys |
| Mirror, lesion, restoration, replay, perturbation, holdout | `steering_benchmark.py` condition runners | Positive/null and seeded confinement tests | 27 named conditions with digests and effects |
| Graph immutability and sparse execution | CSR digest gates in `steering_benchmark.py` | Synthetic and real graph gates | `graph_unchanged=true`, 6,240,402 sparse edges |
| Provenance-first atomic publication | `biological-steering` CLI command | `test_biological_steering_cli.py` failure atomicity tests | Result identity manifest and SHA-256 |
| Legacy claim boundary | `EmbodiedEpisodeResult` literals | `test_embodied_cli.py` | `arbitrary-smoke-only`, `behavior_claim_allowed=false` |
| Full retained snapshot execution | Real-data pytest gate | `test_biological_steering_real.py` | 166,606-neuron retained result audited above |

The audit proves this biological-steering phase, not the broader goal of a complete or intelligent
digital fly. Those stronger claims remain explicitly unachieved.
