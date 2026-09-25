# Motor/body temporal diagnostic — 2026-09-25

The autonomous MaleCNS-to-FlyGym pathway still does not demonstrate learned
food seeking, threat avoidance, or animal-equivalent intelligence. This note
examines a single seed-7, 20-step FlyGym episode without changing its controller.

## Reproduction and provenance

Run from source revision `248a587dd90a50e8a511efb992e61ea37ae47419`:

```bash
PYTHONPATH=src .venv/bin/flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 --steps 20 --seed 7 --backend flygym \
  --motor-trace \
  --output artifacts/autonomous-hexapod-flygym-motor-trace-248a587-seed7-steps20.json
```

The [artifact](../../artifacts/autonomous-hexapod-flygym-motor-trace-248a587-seed7-steps20.json)
has SHA-256 `8ad4659442731bd2f20f0fda83bfdc89542a8e214046fb7c7d014b8e866ac5b7`.
Its `software_revision` is the clean commit above. The optional `motor_trace`
records 36 normalized activations in `CANONICAL_MOTOR_GROUPS` order, decoded
18-joint moments, moments actually sent to the backend after delay/damage,
and the resulting thorax pose/support state at each body step. With the flag
omitted, this trace is `null`; it changes neither the neural path nor the
default trace digest. Replay includes the optional trace in its equality check.

## Observation

The episode produced 464 motor spikes and 20 body steps. Replay was exact,
the retained graph was unchanged, the body did not fall, and there were no
food or threat contacts. Final food distance was `0.0009338525802978096 m`.
Thorax coordinates changed from step 1 `(0.000500261, 0.000000364) m` to
step 20 `(0.000571266, 0.000097637) m`; this includes early physical settling,
so it is not a measure of purposeful advance.

For each antagonistic pair at each leg and step, the cancellation fraction is
`sum(2 * min(a, b)) / sum(a + b)` over its normalized activations. The result
is 0.558 for thorax–coxa, 0.0805 for trochanter, and 0.00852 for tibia.
With both activations above 0.1, pair counts were respectively 54, 2, and 0
out of 120 leg-step pairs per joint. This is a decoder-drive cancellation
proxy, **not** measured muscle force or proof that coactivation caused the
poor trajectory. The adapter still actuates only 18 of FlyGym's 66 leg DOFs,
with no adhesion control. A separate 100-step run from the previous source
revision also showed no useful sustained food approach.

## Integration-test boundary

With `FLYBRAIN_MALECNS_SNAPSHOT` set, the new real two-step motor-trace test
passes. Two older tests expecting a positive dopamine effect in only two
reference-body steps fail on the current contact-gated-neural-DAN configuration.
The observed reference episode had two appetitive contacts and 13 distinct
spiking declared DAN IDs, but zero *contact-coincident recruited* DAN spike
events and therefore zero slow-memory dopamine effect. The old positive
assertions are not evidence that the optional trace broke learning. The
normal test suite, which skips retained-snapshot tests without the explicit
environment variable, passed 448 tests and skipped 18.

## Next causal gate

Use the recorded activation and torque time series to test whether silencing
the measured antagonistic thorax–coxa populations changes gait, forward
displacement, and support under matched seeds and a zero-torque control.
Any intervention must remain an explicit biological lesion/diagnostic, not
a hidden gait policy. Independently, lengthen contact-gated learning episodes
until recruited DAN spiking is actually observed before testing whether
plasticity changes food and threat behavior against lesions and rewired controls.
