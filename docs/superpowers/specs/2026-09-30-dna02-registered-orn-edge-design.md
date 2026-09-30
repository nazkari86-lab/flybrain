# Registered ORN input to DNa02 direct thoracic edges: locked replication

## Scope and provenance

This is an isolated, 200 ms Shiu/MaleCNS simulation with no body, muscle,
plasticity, reward, or behavior. The registered ORN populations and fixed model
parameters come from `artifacts/odor-mbon-probe-aab0371-seeds7-8-9-steps2000.json.gz`
(uncompressed SHA-256
`3f3493acb4b61fa8cedb2f5cc8cb07c84b4367335161e2a9a7045c7c35f3864e`).
The 17 DNa02-to-thoracic-`vnc_motor` edges and 17 same-source
`vnc_intrinsic` controls come from
`artifacts/dna02-direct-edge-isolated-seeds25-27.json.gz` (uncompressed SHA-256
`3d2dc99233197e45d2849aa6b52220689e2b230fcb53f85bce991ac1ba8b0dc5`).
The retained snapshot SHA-256 is
`9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f`.
These source populations are task-assigned odor channels, not measured real
odorants. No source afferents are removed and DNa02 receives no direct
external stimulation.

## Prior observations and status

An exploratory readout of the original ORN seed-7-to-9 panel found DNa02
spikes in all six food/threat conditions, with exact agreement to the
published full-neural-trace digests. Exploratory seeds 34–36 showed direct-edge
target reductions under both ORN inputs, but the same-source matched-control
edge deletion changed DNa02's own spike timing in all three food conditions.
The strict matched-control gate announced before those food outcomes failed
0/3. The threshold below is chosen after those exploratory observations, so
the new panel is a transparent replication check, not an independent
hypothesis-discovery panel.

## Frozen new-seed protocol

Run previously unused seeds 37, 38, 39 for each of food-labeled ORN_DM1/VA2
and threat-labeled ORN_DA2. For each seed/odor, generate 2,000-step Poisson
voltage events once, with `dt_ms=0.1`, `refractory_ms=2.0`,
`synaptic_delay_ms=1.0`, and all other Shiu defaults. Replay those exact ORN
events on the same immutable full graph under: intact; zeroing exactly the 17
frozen direct DNa02-to-motor edges; zeroing exactly the 17 frozen same-source
control edges; no-source; and intact replay. The no-source condition has no
external events. Do not zero any afferents or add stimulation to DNa02.

For a **narrow model-circuit replication**, every one of six seed/odor pairs
must satisfy all of these gates:

1. ORN event digests are equal across intact, direct-edge-zero, and
   control-edge-zero conditions; DNa02 and its 17 targets fire above no-source.
2. The observed DNa02 spike-time digest is identical between intact and
   direct-edge-zero, and the summed spikes of all other `vnc_motor` cells are
   identical in that pair.
3. Direct-edge-zero reduces the 17 target-cell spike total by at least 5%
   relative to intact.
4. No-source has zero DNa02 and target spikes; intact replay matches every
   recorded neural readout; the canonical graph digest is unchanged.

Record, but do **not** use to rescue the primary gate, whether the matched
control preserves the DNa02 spike-time digest and its target-cell reduction.
The control is matched for presynaptic cell, edge count, and approximate edge
weight, not for postsynaptic connectivity or activity. A changed source trace
makes its contrast confounded. Keep both the primary narrow result and the
strict matched-control result visible, even if either fails.

## Claim boundary

Passing can support only a causal contribution of these fixed direct edges
to thoracic motor-neuron firing under registered artificial ORN drive in one
Shiu/MaleCNS model. It does not establish natural odor responses, specific
muscle force, walking, learned behavior, autonomous intelligence, or a
population effect in real flies. The prior closed-loop DNa02 gate failure and
the control-confounded 34–36 outcome remain negative evidence.
