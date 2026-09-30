# Locked same-postsynaptic controls for intact ORN→DNa02→motor assay

## Purpose and selection history

The [registered-ORN DNa02 assay](../../data/dna02-registered-orn-direct-edge-2026-09-30.md)
showed a narrow 6/6 direct-edge effect under artificial ORN input, but its
17 same-source `vnc_intrinsic` control edges fed back into DNa02 and changed
its spike timing in 6/6. This new assay tests whether the direct-edge effect
is distinguishable from lesions of *other excitatory inputs onto the exact
same 17 thoracic `vnc_motor` cells*, without removing DNa02 afferents or
stimulating DNa02 directly. It remains a model experiment, not a living-fly
or learned-behavior experiment.

Candidate edges were selected using the already published, therefore
exploratory, seed-37-to-39 intact ORN runs. For each frozen DNa02 target,
consider positive canonical edges from `vnc_intrinsic` sources to that same
post cell. For each candidate, form a six-element vector of canonical edge
weight × observed presynaptic spike count across food/threat × seeds 37–39.
For the 17-edge control, minimize mean absolute vector difference from the
DNa02 direct edge; break ties by more nonzero source panels, smaller absolute
edge-weight difference, then lower source ID. For the 34-edge control, choose
the distinct pair per target minimizing the same mean absolute error; break
ties lexicographically by source IDs. This selection uses no seed 40–42
outcomes. It estimates input quantity, **not** time-resolved synaptic efficacy.
The single-edge and two-edge controls had about 79% and 91% of the DNa02
aggregate weight×spike exposure on the selection panel, respectively; they
are not perfectly matched, and the two-edge arm intentionally deletes twice
as many edges. Neither arm shares the DNa02 presynaptic identity.

## Frozen source-to-post pairs

The posts below are the already frozen 17 thoracic motor targets. The runner
must verify every listed positive edge, source superclass, exact graph
position and weight against the retained snapshot. No candidate may be
replaced after seed 40–42 outcomes are seen.

| Motor post ID | Single VNC-intrinsic source | Two-source VNC-intrinsic control |
| ---: | ---: | ---: |
| 800158 | 802450 | 802450, 811021 |
| 800316 | 809067 | 807487, 809067 |
| 801469 | 906407 | 811006, 906407 |
| 801813 | 800373 | 800373, 810066 |
| 801918 | 906407 | 800214, 906407 |
| 801946 | 904800 | 801421, 904800 |
| 906235 | 800846 | 800846, 807487 |
| 801548 | 800383 | 800383, 801550 |
| 802215 | 800631 | 907028, 909475 |
| 804851 | 801111 | 801111, 806584 |
| 806923 | 803035 | 803035, 804647 |
| 815344 | 804217 | 800173, 904395 |
| 818399 | 801149 | 801149, 802757 |
| 832490 | 903653 | 804773, 903653 |
| 903689 | 802780 | 802780, 905387 |
| 919509 | 803035 | 800075, 803035 |
| 927808 | 800747 | 800174, 803210 |

## Prospective test

Use previously unused seeds **40, 41, 42** and both registered ORN source
populations from the hash-verified prior artifact. For each odor/seed,
generate one Poisson event dictionary and reuse it in intact, DNa02 direct
17-edge zero, same-post single 17-edge zero, same-post pair 34-edge zero,
no-source and exact intact replay. Keep 2,000 Shiu steps and all parameters
from the previous assay. All interventions are sparse zero multipliers on
the immutable MaleCNS graph. Record event digest, DNa02 spike-time digest,
the spike-time digest of all selected control presynaptic cells, 17 target
counts, other `vnc_motor` spikes, whole trace digest, replay, graph digest,
runtime, peak RSS and all input/software hashes.

The **strict same-post control gate** requires every one of the six new
odor/seed pairs to meet all conditions:

1. Identical ORN event digests; DNa02 and targets recruited above no-source;
   no-source has zero DNa02 and target spikes.
2. Identical DNa02 **and selected control-source spike-time digests** across
   intact and all three edge-zero conditions. The aggregate spike count of
   other `vnc_motor` cells is also identical across all four conditions.
3. Direct-edge zero reduces 17-target spikes by at least 5% and by at least
   **5 percentage points more than both** same-post control arms.
4. In the intact run, the two-edge control's summed canonical
   weight×presynaptic-spike exposure is between **80% and 120%** of the
   DNa02 direct edges' analogous exposure. This is a model-input balance
   check, not physiological equivalence.
5. Intact replay is exact and the canonical graph digest is unchanged.

Report all thresholds unchanged even if the gate fails. A positive gate
would support stronger *model-level* DNa02 edge selectivity under registered
ORN stimulation. A failure would not erase the previous narrower direct-edge
causal result. Neither result proves natural odor perception, specific muscle
force, gait, associative learning, autonomous intelligence, or a population
effect across real flies. The previous closed-loop DNa02 gate remains failed.
