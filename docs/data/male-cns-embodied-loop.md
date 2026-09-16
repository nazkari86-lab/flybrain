# MaleCNS embodied-loop smoke

Measured on 2026-09-16 with the complete retained MaleCNS v1.0 graph and seed 7.

## Identity and protocol

- Dataset: `male-cns-v1.0-essential`
- Source manifest SHA-256: `8406eacdb75db4f1b84cfbf13b281b51628e68941458504312c0f42329fd5122`
- Snapshot content SHA-256: `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`
- Graph: 166,606 neurons and 6,240,402 directed edges
- Protocol: 100 world steps, 100 neural steps, seed 7
- Interface: 32 declared sensory neurons and 16 declared motor neurons

## Measured result

The run passed deterministic replay and both canonical/executed graph-integrity gates. One of the
100 decoded commands was non-zero. The body moved from `x=5.0` to `x=5.001875`; final forward
speed was `0.01875`, and energy was `0.9999875`. Runtime was 0.322 seconds and peak RSS was
1,026,621,440 bytes on this host. The committed software revision recorded by the artifact is
`f7acd6a81a611a3242926cafc5dac4b27d9445e8`.

The machine-readable result is `artifacts/embodied-malecns-provenance-seed7.json` in the local
artifact store. Its full traces include sensory events, actions, rewards, dopamine, and body state.

## Scientific boundary

This is proof that the full sparse MaleCNS dynamics can participate in a deterministic
world-to-neuron-to-motor-to-body feedback loop. The selected populations are currently bounded,
declared subsets of role annotations rather than validated visual and motor cell-type maps. The
single small movement is therefore a positive integration result, not food-seeking, navigation,
animal-level intelligence, or biological fidelity. Those claims require subtype-specific sensory
maps, opponent motor populations, longer trials, controls, lesions, and out-of-sample behavior.
