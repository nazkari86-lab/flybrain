# Autonomous generalization protocol v4 — 2026-09-23

## What changed

Protocol `measured-replay-persistent-memory-v4` adds a fail-closed world-split gate.
`world_digest` hashes food and threat geometry, initial body position and physical
perturbation while excluding the variant name and evaluation label. Renaming a
training world can no longer make it look unseen.

A behavioral claim now additionally requires:

- no world digest shared between training and holdout;
- at least two distinct holdout worlds scored for food;
- at least two distinct holdout worlds scored for threat;
- the existing three independent replicates, positive lower 95% intervals against
  every control, exact replay, immutable graph and frozen holdout memory.

The retained defaults evaluate four holdout worlds. The second food world mirrors
source geometry, starts at `(0.02, -0.01)`, uses friction scale `0.9`, mass scale
`1.05`, and one motor-delay step. The second threat world mirrors source geometry,
starts at `(-0.01, 0.02)`, uses friction scale `1.1`, and mass scale `0.95`. These
world parameters are shared by normal and every control and never enter the neural
controller.

## Reproduction

```bash
./.venv/bin/flybrain experiment autonomous-behavior \
  artifacts/male-cns-v1.0-w5 \
  --output artifacts/autonomous-behavior-generalization-v4-seeds7-11-13-steps2-corrected.json \
  --training-episodes 1 --holdout-episodes 1 --steps 2 --seeds 7,11,13
```

## Measured retained result

| Field | Value |
| --- | --- |
| Artifact SHA-256 | `376d7ee227f91dbbedc4ff4c84ec602684be03cac59da9f46374431097670b4c` |
| Snapshot SHA-256 | `9b7d4594216c8b37b5ffc4199f2b5cd448a77d0b1daadfbb3de8b322738a2f1f` |
| Graph | 166,606 neurons; 6,240,402 directed edges |
| Seeds | `7, 11, 13` |
| Runtime / peak RSS | `109.2049 s / 1,205,436,416 bytes` |
| Episode evidence | 90 primary episodes; 90/90 exact replay |
| Frozen evaluation | 60/60 complete learning-memory holdouts |
| World split | 2 training; 4 holdout; 0 overlap; food 2; threat 2 |
| Unassisted motor output | true; external phase amplitude `0.0` |
| Graph unchanged | true |
| Behavioral claim | false |

Normal training recorded six appetitive contacts, six aversive contacts and twelve
contact-routed DAN events across seeds. Normal holdouts produced motor activity, but
all food and threat deltas against no-plasticity, DAN lesion, KC→MBON lesion and
rewired control were exactly `0.0` with intervals `[0.0, 0.0]`. Two body steps are
only a protocol smoke horizon; no meaningful navigation can be inferred.

This corrected artifact uses independent `SeedSequence` episode streams, preserves
the declared π/2 threat-plume phase in bilateral sensing, and uses mirrored
thorax–coxa torque signs. The earlier same-day v4 artifact without the `corrected`
suffix predates those review fixes and is historical only.

## Interpretation boundary

`generalization_verified=true` means only that the declared evaluation worlds are
structurally unseen and sufficiently diverse for this gate. It does not mean the
agent succeeded in them. This run proves world-split integrity, deterministic
replay, frozen memory, control execution and graph immutability. It does not prove
learned food seeking, threat avoidance, broad transfer, animal equivalence,
consciousness or complete autonomous intelligence.
