# Retained MaleCNS visual looming projection assay

## Scope

This assay resolves exact retained MaleCNS LC4, LPLC2, DNp01, and DNp02 populations and
tests whether an anonymous looming projection stimulus can recruit endogenous descending
neurons. It uses the immutable 166,606-neuron, 6,240,402-edge graph, Shiu dynamics, fixed
lesions, exact replay, and a graph-integrity gate.

The stimulus is deliberately injected at the LC4/LPLC2 projection interface. Upstream retina,
lamina, medulla, and optic-lobe processing are bypassed and the artifact records
`upstream_visual_processing_bypassed=true`. Therefore this is a causal projection-path assay,
not a complete retina-to-behavior experiment.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/flybrain experiment visual-looming \
  artifacts/male-cns-v1.0-w5 \
  --registry data/registry/biological-interface-registry-v2.json \
  --steps 40 --seed 7 \
  --output artifacts/visual-looming-malecns-v2-seed7.json
```

The output is intentionally not overwritten by the CLI. The retained artifact is:
`artifacts/visual-looming-malecns-v2-seed7.json`.

## Immutable identities

| Field | Value |
| --- | --- |
| Dataset | `male-cns-v1.0-essential` |
| Registry | `male-cns-biological-interface-v2` |
| Registry SHA-256 | `888e6daef16553a410d7b0fc930761e0b5cf39a08e6ea98c9477adf9738dc80f` |
| Snapshot content SHA-256 | `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f` |
| Annotation SHA-256 | `ead87b4d37f97968f2845d525309649a1db9c7e8a032b147af32d51318eaa67b` |
| Result SHA-256 | `4788f05e0fc3bed083b5e222230ef01ec3b15ac1459f8d47d1f225b91ce9e3dc` |
| Protocol | `visual-looming-projection-assay-v1` |
| Seed / steps | `7 / 40` |

The exact retained interface counts are LC4 left/right `71/55`, LPLC2 left/right `94/91`,
and one exact DNp01 and DNp02 neuron per side. Direct retained connectivity measured for the
same graph is LC4→DNp01 `126` edges, LC4→DNp02 `124`, and LPLC2→DNp01 `176`.

## Measured conditions

| Condition | DNp01 spikes | DNp02 spikes | Interpretation |
| --- | ---: | ---: | --- |
| Static | 0 | 0 | No projection drive |
| Looming | 2 | 2 | Endogenous descending recruitment |
| LC4 lesion | 2 | 0 | DNp02 response removed in this protocol |
| LPLC2 lesion | 2 | 2 | No measured loss under this coarse input |
| Joint LC4+LPLC2 lesion | 0 | 0 | Both responses removed |
| DNp01 lesion | 0 | 2 | Downstream population specificity control |
| Looming replay | 2 | 2 | Exact deterministic replay |

The result has `causal_pathway_claim_allowed=true`, `replay_exact=true`, and
`graph_unchanged=true`. Every condition retains the explicit upstream-bypass marker.

## Scientific boundary

This evidence supports a causal claim that the declared retained projection populations can
recruit DNp01/DNp02 under the stated interface stimulus. It does not establish complete
retina-to-optic-lobe processing, motor recruitment, learned threat avoidance, generalization,
animal-equivalent behavior, or autonomous intelligence. `behavioral_claim_allowed` remains
`false` until multi-seed holdout behavior beats the declared no-plasticity, lesion, rewired,
random, and unseen-world controls.
