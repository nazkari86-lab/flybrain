# Isolated odor-to-MBON recruitment probe

## Goal

Make the exploratory odor-to-MBON bottleneck measurement reproducible on the
retained MaleCNS snapshot without changing the autonomous controller or using
it as behavioral evidence.

## Boundary and data flow

A standalone diagnostic reads the existing snapshot, learning registry,
KC→MBON manifest and measured-side olfactory receptor map. It runs
`poisson_voltage_events` into `simulate_shiu` on the immutable graph with a
fresh state for each condition. No body, reward, DAN update, visual input,
policy or target coordinate enters this probe. The probe is separate from the
closed-loop benchmark.

The initial fixed panel uses seeds 7, 8 and 9; 2,000 neural steps; the
existing 0.1 ms, 2 ms refractory, 1 ms delay and 150 Hz source-equivalent
parameters; both sides of food ORN_DM1/ORN_VA2 and threat ORN_DA2. For each
seed, compare no odor, food, threat, and both with baseline overlay, plus
threat with all declared KC→MBON 519128 multipliers at the allowed 2.0 upper
bound. Reuse the exact threat event train between baseline and upper-bound
conditions. The 2.0 condition is labeled an artificial intervention, never
learning.

## Output and gates

Write one JSON result with snapshot/registry identity, source revision,
parameters, source IDs/counts, per-condition source events and spikes,
plastic-KC input spikes, target MBON spikes, peak post-step voltage,
signed presynaptic weighted-spike sums, DAN route counts, graph identity,
and an exact baseline replay check. The output always states
`autonomous_behavior_claim_allowed=false` and makes no claim about embodied
learning. The CLI rejects invalid seeds/steps, absent receptor groups or
target IDs, and existing output paths; writes atomically without overwriting
inputs.

## Verification

Unit tests on a small signed graph must show that changing a declared
KC→MBON overlay changes the probe readout without mutating the graph, that
source-free controls stay inactive, and that replay is exact. A retained
snapshot smoke test checks real receptor selection and schema. Fresh Ruff,
mypy, relevant pytest, data hash checks and the remote-publication check
precede any success statement.
