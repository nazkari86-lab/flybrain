# Embodied loop fixture

The first closed-loop fixture is intentionally small and deterministic. It validates the wiring
between a sparse graph, world observations, sensory voltage events, motor decoding, and body state
feedback without claiming a biologically calibrated behavior.

## Run

```bash
uv run flybrain snapshot import-csv \
  --neurons tests/fixtures/tiny/neurons.csv \
  --edges tests/fixtures/tiny/edges.csv \
  --output artifacts/embodied-fixture-snapshot \
  --dataset-id tiny-v1 \
  --manifest-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

uv run flybrain experiment embodied-loop artifacts/embodied-fixture-snapshot \
  --max-steps 8 \
  --output artifacts/embodied-loop-fixture.json
```

The machine-readable result is `artifacts/embodied-loop-fixture.json`. It records the selected
neuron maps, every sensory event, action, reward, body state, replay digest, and software revision.
The fixture demonstrates deterministic sensory feedback and safe bounded execution. Its eight-step
window is shorter than the reference Shiu synaptic delay, so zero motor output is an expected null
result for this particular tiny graph; the longer episode tests use a synthetic circuit to verify
motor causality and body motion.

The world is a research harness, not a validated fly body. Food and threat contact currently
produce observable reward signals, but calibration of sensory tuning, motor populations, muscles,
proprioception, and learned food-seeking remains a later scientific benchmark.
