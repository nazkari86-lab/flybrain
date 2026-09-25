# Contact-gated DAN learning horizon — 2026-09-25

The objective is learned, autonomous behavior, not just nonzero plasticity.
This diagnostic isolates a prerequisite: whether real registered appetitive
DAN spikes can coincide with a food contact in the retained MaleCNS simulation.

At two reference-body steps with seed 7, food contact occurred twice, but
all 20 observed DAN spikes came from 13 **aversive** neuron IDs. None came
from the 316 registered appetitive DAN IDs. Consequently the strict
`contact_gated_neural_dan` route selected no DAN spikes, and dopamine/NO
effects remained zero. Directly recruiting all appetitive DANs from contact
would hide this boundary and was not used.

The same source revision `6aa065a546c57090388a442bae31e337196bc480`,
snapshot, seed and reference backend were run for 20 body steps:

```bash
PYTHONPATH=src .venv/bin/flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 --steps 20 --seed 7 --backend reference \
  --output artifacts/autonomous-hexapod-reference-contact-gated-6aa065a-seed7-steps20.json
```

The [clean-revision artifact](../../artifacts/autonomous-hexapod-reference-contact-gated-6aa065a-seed7-steps20.json)
has SHA-256 `08915930bec0cd2907ec8d856467b11ee7bb2daf3842c9fabe8479c18e5bbed3`.
It records 20 food contacts, zero threat contacts, 318 observed appetitive
DAN spikes, 538 observed aversive DAN spikes, 18 windows with recruited
appetitive DAN activity, maximum dopamine effect `4.058649967912646e-05`,
maximum nitric-oxide effect `4.555035841785069e-07`, and 9,678 of 33,496
plastic multipliers different from one. Replay was exact and the canonical
graph was unchanged. The initial food distance is zero, so these contacts
are **not** evidence of food-seeking behavior.

Restoring the resulting durable memory and running another 20-step episode
with the same seed increased the maximum dopamine effect to
`0.0001655466031324268`; the second episode had 18 recruited-DAN windows,
20 food contacts, and exact replay. The real-data memory regression test now
requires this sufficient horizon. The separate two-step CLI test checks
atomic artifact publication and the strict reinforcement source, not an
unjustified positive dopamine effect before appetitive DANs fire.

This establishes that the declared neural-DAN path **can** induce and retain
plasticity in a persistent-contact reference setup. It does not show learned
navigation, transfer to FlyGym, conditioned preference, threat avoidance,
or general autonomous intelligence. The next behavioral requirement remains
better food approach and threat avoidance than no-plasticity, DAN-lesion,
KC-to-MBON-lesion and structural-rewiring controls on independent holdouts.

## Retained MBON-to-motor audit after the 36-group expansion

The full MaleCNS integration suite initially retained two anatomical test
expectations from the older motor registry. The current v2 registry contains
182 motor IDs, including all 148 from v1. The exact retained graph has zero
direct MBON-to-motor edges, but 60 distinct second-hop edges, 108 two-hop
routes through 16 intermediates, and 1,502 second-hop synaptic contacts.
The original 31 edges, 46 routes and 376 contacts remain within that total;
the additional motor IDs account for 29 edges and 62 routes. The full route
digest is `44f4383a89f1be558016e024e1fbce8c25cc49beb0db82ce750e52cc917e8e55`.

In the separate *scheduled* associative-motor probe, learned versus baseline
motor spike counts were 71 versus 70 after paired training; the delayed,
unpaired control differed by -12 spikes. The paired probe had 33 fewer MBON
spikes, exact replay, and an unchanged graph. Persistent neural activity
still blocks a behavioral claim, and a one-spike motor difference under a
predeclared schedule is not evidence of autonomous learned navigation.
