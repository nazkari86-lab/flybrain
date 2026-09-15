# Mushroom-Body Dopamine Plasticity Design

## Goal

Add persistent, cue-specific associative memory on the real MaleCNS Kenyon-cell to mushroom-body-output-neuron connectivity, without training a separate policy network or changing anatomical topology.

## Biological scope

MaleCNS v1.0 contains 4,064 neurons annotated as `Kenyon_Cell`, 340 as `DAN`, 97 as `MBON`, and 33,496 retained KC→MBON edges at the current `weight >= 5` threshold. Dopamine-dependent coincidence at KC→MBON synapses is a central mechanism of fly associative learning, but the released structure does not provide every compartment-specific receptor and plasticity coefficient.

The implementation therefore separates evidence and assumption:

- measured: neuron identity, KC→MBON topology, synapse counts, transmitter annotations;
- literature-grounded mechanism: dopamine gates plasticity only at recently active KC→MBON synapses;
- explicit fitted assumptions: eligibility decay, learning rate, and lower/upper weight bounds.

## Rule

Each plastic edge keeps baseline weight `w0`, current multiplier `m`, and eligibility `e`:

```text
e <- exp(-dt / tau_e) * e + pre_activity * post_gate
m <- clip(m - eta * dopamine * e, min_multiplier, max_multiplier)
```

Positive dopamine depresses eligible KC→MBON synapses, matching the commonly observed depression mechanism in conditioned mushroom-body compartments. The API allows signed modulation for future compartment-specific appetitive/aversive rules, but benchmark claims are limited to cue-specific synaptic memory rather than behavioral valence.

## Real-topology benchmark

1. Select the real MBON with the largest number of retained KC inputs.
2. Deterministically choose two disjoint connected KC ensembles: cue A and cue B.
3. Measure both weighted input responses.
4. Pair cue A eligibility with dopamine for repeated trials; never activate cue B during reinforcement.
5. Verify cue A response decreases while cue B remains unchanged.
6. Run controls with dopamine disabled and with eligibility cleared before dopamine.
7. Save and reload plastic state; recall must remain byte-stable at the observable weight layer.

## Acceptance

- Fixture tests prove locality, bounds, decay, no-dopamine control, eligibility timing, and persistence.
- Real MaleCNS benchmark processes all 33,496 retained KC→MBON edges.
- Trained-cue response changes by at least 10%; untrained-cue relative drift stays below `1e-6`.
- No-dopamine and cleared-eligibility controls show zero learned change.
- The benchmark reports source populations, selected target MBON, cue IDs, parameters, before/after responses, controls, runtime, memory, and state checksum.

This phase proves persistent associative synaptic memory. It does not yet prove action selection, reward discovery, or embodied intelligence; those require closed-loop sensory and motor interfaces.
