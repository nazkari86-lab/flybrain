# Shiu Whole-CNS Dynamics Smoke Run

## Provenance

The defaults reproduce constants and equations in the primary reference implementation
[`philshiu/Drosophila_brain_model/model.py`](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py):

| Parameter | Value |
|---|---:|
| time step | 0.1 ms |
| resting/reset potential | -52 mV |
| spike threshold | -45 mV |
| membrane time constant | 20 ms |
| conductance decay | 5 ms |
| refractory period | 2.2 ms |
| synaptic delay | 1.8 ms |
| per-synapse amplitude | 0.275 mV |
| Poisson sensory rate | 150 Hz |
| Poisson input scale | 250 |

The implementation uses the analytic solution of the coupled membrane/conductance equations for
each 0.1 ms step. Spike propagation walks only fired presynaptic CSR rows and queues their effects
for the exact 18-step delay. As in the source model, Poisson targets are exempt from refractory
time. No policy network, game coordinates, or authored motor rule participates.

## Measured full-graph run

Command:

```bash
/usr/bin/time -l uv run flybrain experiment shiu-smoke \
  artifacts/male-cns-v1.0-w5 --duration-ms 10 --seed 7 \
  --output artifacts/shiu-smoke-10ms-seed7.json
```

| Metric | Value |
|---|---:|
| neurons | 166,606 |
| canonical edges | 6,240,402 |
| unresolved-sign canonical edges | 1,177,887 |
| stimulated sensory neurons | 17,336 |
| Poisson voltage events | 25,926 |
| total emitted spikes | 32,344 |
| distinct reached neurons | 18,951 |
| sensory spikes | 25,934 |
| interneuron spikes | 5,264 |
| ascending spikes | 685 |
| descending spikes | 273 |
| motor spikes | 188 |
| simulator runtime | 5.10 seconds |
| independent maximum RSS | 1,021,984,768 bytes |
| outgoing event CSR | 76,217,680 bytes |

Activity digest:

```text
58d427a318191159ff01f5c1abffe26beb7f85968f08e68a7d30d50c1868459f
```

A fresh second run with the same seed reproduced exact neuron/edge counts, input-event count, total
spikes, reached-neuron count, per-role spike counts, parameters, and activity digest. Its runtime
was 4.88 seconds and peak RSS 1,035,927,552 bytes; resource timing is intentionally excluded from
the deterministic contract.

## Interpretation and remaining limitations

This result proves that measured sensory activity propagates through the full retained MaleCNS
topology into ascending, descending, and motor populations under a published LIF model. It does not
prove that the resulting motor activity is behaviorally correct. The model still lacks
receptor-specific glutamatergic signs, explicit neuromodulator kinetics, morphology-derived delays,
plasticity, muscles, proprioception, and closed-loop environmental feedback.
