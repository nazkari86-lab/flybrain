# Embodied sensory-motor loop design

## Goal

Add the first genuinely closed-loop body and world to FlyBrain. The result must let the measured
connectome receive world-derived sensory events, emit motor events, change a simulated body, and
feed body state back into the next neural step. It is a research instrument, not a claim that the
model is a living fly.

## Scope of this stage

This stage uses a deterministic planar arena so every causal quantity is inspectable and tests do
not depend on an external physics engine. The world contains a fly body, food, a threat, and a
wall. The body has position, heading, forward speed, angular speed, energy, and six leg contact
states. The body integrates bounded motor commands with semi-implicit Euler updates and resolves
wall collisions deterministically.

The sensory interface exposes luminance-like visual sectors, odor concentration, contact, wind,
and proprioception. Encoders convert these continuous observations into timestamped voltage events
for declared sensory neuron IDs. No CNN, object detector, scripted obstacle avoidance, policy
network, or hard-coded `go_to_food` action is allowed.

The motor interface reads only declared motor/descending neuron populations and applies a fixed,
versioned decoder to left/right turn and forward thrust. It reports the decoder mapping and every
command so causal audits can distinguish model output from world integration.

## Architecture

`ArenaWorld` owns immutable arena objects and mutable `FlyBody`; `SensoryEncoder` observes the
world and emits `ExternalEvent`s; `run_embodied_episode` advances Shiu dynamics in bounded
chunks; `MotorDecoder` converts emitted spikes into a command; the world integrates the command;
the next chunk observes the resulting body state. Reward is an observation of the world outcome,
not a direct action instruction: food contact gives positive dopamine and threat contact gives
negative dopamine. The existing plasticity API receives only the resulting neuromodulator signal.

The episode runner stores event, action, reward, body, and neural summary traces. It supports a
baseline and a learned connectome, deterministic replay, an intervention that silences a selected
output population, and an explicit null/causal acceptance result.

## Boundaries and safety

- Canonical connectome topology remains immutable; only the existing plastic overlay may change.
- World physics never reads internal neuron state to choose an action.
- The runner never injects a target-directed motor command.
- All IDs are validated against the snapshot before the episode starts.
- Episode length, command magnitudes, energy, and arena bounds are bounded.
- Unsupported sensory/motor mappings fail loudly instead of silently inventing IDs.

## Acceptance gates

1. A tiny synthetic snapshot completes at least one episode with non-empty sensory events and a
   motor trace.
2. A world perturbation changes the next sensory event digest while replay with the same seed is
   byte-for-byte identical.
3. A wall collision keeps the body inside the arena and reduces forward progress.
4. Reward is emitted only after the corresponding world contact; no contact means zero dopamine.
5. Silencing the declared motor output population changes the action trace on a fixture circuit.
6. The full runner preserves the canonical graph digest and reports all provenance and traces.

This stage does not claim successful food seeking or threat avoidance. Those are later behavioral
benchmarks requiring calibrated sensory maps, richer motor decoding, and enough validated biology.
