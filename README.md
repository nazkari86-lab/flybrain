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
