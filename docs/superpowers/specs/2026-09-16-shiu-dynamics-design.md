# Shiu Reference Dynamics Design

## Goal

Run source-traceable whole-CNS spiking dynamics on the canonical MaleCNS snapshot using the published Shiu et al. model constants and equations, without a dense matrix or a hidden policy network.

## Source model

The reference implementation in `philshiu/Drosophila_brain_model/model.py` defines:

- resting and reset potential: `-52 mV`;
- spike threshold: `-45 mV`;
- membrane time constant: `20 ms`;
- alpha-synapse conductance decay: `5 ms`;
- refractory period: `2.2 ms`;
- synaptic delay: `1.8 ms`;
- per-synapse amplitude: `0.275 mV`;
- sensory activation baseline: Poisson `150 Hz`, direct voltage jump scaled by `250`.

The simulator records these values as an immutable parameter object. The coupled linear membrane/conductance equations use their analytic discrete-time solution rather than Euler drift. Refractory neurons hold reset voltage and zero conductance, matching the source equations marked `unless refractory`.

## Event graph

The canonical post-by-pre CSR remains the analysis representation. A one-time transpose creates pre-by-post outgoing CSR for event propagation. Each spike walks only its outgoing row and accumulates delayed conductance into a ring buffer. Complexity per step is proportional to active outgoing edges, not all 6.24 million edges.

Zero-sign canonical edges remain present in provenance metrics but contribute zero instantaneous conductance. They will become active only under later receptor-aware or neuromodulatory rules.

## Determinism and intervention

Named seeds generate Poisson sensory events. State includes voltage, conductance, refractory counters, delayed-event ring, step, and RNG state. Silencing suppresses spike emission while preserving incoming subthreshold state. Checkpoint support is deferred until the reference state passes analytic and full-graph tests.

## Acceptance

1. Analytic single-neuron decay matches a hand-derived solution.
2. A two-neuron spike arrives after exactly `1.8 ms` at `dt=0.1 ms`.
3. Refractory behavior prevents re-spiking for `2.2 ms`.
4. Identical seeds produce identical Poisson events and spike traces.
5. Full MaleCNS loads into outgoing sparse form and completes a 10 ms sensory-drive smoke run below 8 GiB peak RSS.
6. The run reports stimulated sensory cells, spikes, reached neurons, role distribution, unresolved edges, runtime, and parameters.

This phase validates dynamics and propagation. It does not claim adaptive intelligence until plasticity and closed-loop sensory/body feedback are implemented.
