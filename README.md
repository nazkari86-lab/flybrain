# FlyBrain

FlyBrain is a reproducible, provenance-aware sparse connectome simulation research platform.
The current milestone is deliberately narrow: it verifies data contracts, imports directed sparse
graphs, runs deterministic LIF dynamics, performs paired cell-type lesions, restores checkpoints,
and exports complete experiment bundles.

It is not yet an embodied fly and does not claim to recreate an animal. Membrane parameters,
receptor effects, conduction delays, and learning rules remain modelling assumptions until a later
validated stage supplies evidence for them.

## Setup

```bash
uv sync --extra dev
uv run flybrain --help
```

## Run the tiny causal experiment

Create the canonical snapshot:

```bash
uv run flybrain snapshot import-csv \
  --neurons tests/fixtures/tiny/neurons.csv \
  --edges tests/fixtures/tiny/edges.csv \
  --output artifacts/tiny-snapshot \
  --dataset-id tiny-v1 \
  --manifest-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

The committed example configuration targets that snapshot:

```bash
uv run flybrain experiment run tests/fixtures/tiny/experiment.json \
  --output artifacts/tiny-run
```

Expected causal result: one normal motor spike, zero motor spikes after silencing the relay cell
type, and `lesion_effect = 1.0`.

## Quality gate

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Official source URLs and their current storage status are recorded in
`data/registry/sources.yaml`. Large synapse-coordinate products are deferred while this machine has
only 32 GiB free because downloading all three would violate the mandatory 10 GiB reserve.

The four essential MaleCNS v1.0 products are declared with verified SHA-256 hashes in
`data/manifests/male-cns-v1.0-essential.json`. Acquire or revalidate their local content-addressed
cache with:

```bash
uv run flybrain data acquire data/manifests/male-cns-v1.0-essential.json --root data/cache
```

## Import the real MaleCNS v1.0 graph

The adapter reproduces the official valid-superclass selection and defaults to the published
connection threshold `weight >= 5`. It streams the 151,856,684-row source table instead of loading
it into memory:

```bash
uv run flybrain snapshot import-malecns \
  data/manifests/male-cns-v1.0-essential.json \
  --cache-root data/cache \
  --output artifacts/male-cns-v1.0-w5 \
  --min-weight 5
```

The verified run on this host selected 166,606 neurons and retained 6,240,402 directed edges.
Unknown receptor-dependent and neuromodulatory signs remain explicit as zero-sign edges; they are
not silently converted to excitation. See `docs/data/male-cns-v1.0-import.md` for measured counts,
resource use, and scientific limitations.

## Run published Shiu dynamics on the whole CNS

The source-profile simulator uses `dt=0.1 ms`, membrane time constant `20 ms`, alpha-synapse
decay `5 ms`, refractory period `2.2 ms`, synaptic delay `1.8 ms`, and `0.275 mV` per signed
synapse. All annotated sensory neurons receive deterministic seeded Poisson drive at `150 Hz`:

```bash
uv run flybrain experiment shiu-smoke artifacts/male-cns-v1.0-w5 \
  --duration-ms 10 \
  --seed 7 \
  --output artifacts/shiu-smoke-10ms-seed7.json
```

The measured reference run emitted activity in sensory, interneuron, ascending, descending, and
motor populations. It is a propagation validation, not evidence of adaptive intelligence; learning
and body feedback are separate required stages. See `docs/data/shiu-whole-cns-smoke.md`.

## Run the paired plastic-memory integration

This experiment replays the same two recorded Kenyon-cell cues through baseline and plasticized
Shiu dynamics. It measures cue-attributable target input, preserves provenance, and publishes the
result without overwriting an existing artifact:

```bash
uv run flybrain experiment shiu-plastic artifacts/male-cns-v1.0-w5 \
  --association artifacts/mb-association-seed7.json \
  --state artifacts/mb-association-seed7-state.npz \
  --output artifacts/shiu-plastic-memory-seed7.json
```

The measured run and its scientific limits are recorded in
`docs/data/male-cns-shiu-plastic-memory.md`. This is a dynamic synaptic-memory integration test,
not yet an embodied fly: there is no body, environmental loop, neuromodulatory ecology, or
behavioral validation.

## Run the embodied loop fixture

The deterministic planar embodied harness connects world observations to arbitrary fixture neurons,
decodes motor spikes into bounded body commands, and feeds body state into the next neural step. Its
artifacts are machine-labeled `arbitrary-smoke-only`, with behavior claims disabled:

```bash
uv run flybrain experiment embodied-loop artifacts/tiny-snapshot \
  --max-steps 8 \
  --output artifacts/embodied-loop-fixture.json
```

See `docs/data/embodied-loop-fixture.md`. This validates closed-loop plumbing and replay, not yet
biologically calibrated food seeking or threat avoidance.

The same runner has also passed a measured 100-step smoke test on the full 166,606-neuron MaleCNS
snapshot; see `docs/data/male-cns-embodied-loop.md` for exact identities, resource use, the observed
motor output, and the remaining biological limitations.

## Run the evidence-bound biological steering assay

The biological steering command resolves exact R1-R6, HS, LC16, DNa02, DNg13, and MDN populations
from the retained annotation table before loading or executing the graph. It publishes direct-DN,
open-loop visual, lesion, restoration, replay, holdout, perturbation, and reduced closed-loop
results atomically:

```bash
uv run flybrain experiment biological-steering artifacts/male-cns-v1.0-w5 \
  --registry data/registry/biological-interface-registry-v1.json \
  --steps 40 \
  --seed 7 \
  --output artifacts/biological-steering-malecns-seed7.json
```

The retained MaleCNS run passed every direct calibration and integrity gate. HS feature-path
steering was positive under direct HS stimulation, while photoreceptor and LC16-to-MDN outcomes
were null. Direct HS/LC16 stimulation bypasses upstream visual processing and is not full
retina-to-behavior evidence. See
[`docs/data/male-cns-biological-steering.md`](docs/data/male-cns-biological-steering.md) for exact
hashes, effects, resource use, and limitations.

## Run the evidence-bound hexapod motor assay

The hexapod command resolves 24 exact flexor/extensor motor populations, six proprioceptive banks,
and six descending populations from one registry. It publishes direct motor calibration,
phase-gait calibration, DN-to-motor, proprio-to-motor, and stateful neural-body feedback as
separate result families; it never turns direct calibration into a sensory or locomotion claim.

```bash
uv run flybrain experiment hexapod-motor artifacts/male-cns-v1.0-w5 \
  --registry data/registry/hexapod-motor-registry-v1.json \
  --steps 90 \
  --seed 7 \
  --output artifacts/hexapod-motor-malecns-seed7.json
```

The retained 90-step run calibrated all 24 direct antagonist groups, but phase-gait and all six
DN-to-motor paths were null. Two of six proprioceptive banks passed, and the full feedback loop
was directionally wrong under its explicit mirrored-interface control. These are recorded without
retuning in [docs/data/male-cns-hexapod-motor.md](docs/data/male-cns-hexapod-motor.md). The assay
is a sparse, causally tested motor foundation—not evidence of a walking or intelligent animal.

## Run the autonomous retained hexapod loop

The autonomous command removes the external conditioning schedule. Anonymous olfactory activity
propagates through the retained graph, physical contact recruits registered DANs, sparse local
KC-to-MBON plasticity updates existing positive edges, all 24 motor groups remain available to the
decoder, and six-bank proprioception returns body state to the graph. No reward scalar, target
coordinate, desired action, or hidden policy enters the controller.

```bash
uv sync --extra dev --extra physics
uv run --extra physics flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 \
  --steps 2 \
  --seed 7 \
  --backend reference \
  --output artifacts/autonomous-hexapod-reference-seed7.json
```

Use `--backend flygym` to run the same neural and torque inputs on pinned FlyGym 2.1.0 / MuJoCo
3.9.0. The retained two-step reference and FlyGym runs both preserved the graph, replayed exactly,
produced motor spikes, closed physical contact-to-DAN learning, and returned proprioception. See
[docs/data/male-cns-autonomous-hexapod.md](docs/data/male-cns-autonomous-hexapod.md) for identities,
measurements, reproduction commands, and the remaining behavioral-validation gates.
