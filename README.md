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

## Run the fast practical autonomous fly

This path prioritizes a working autonomous agent over a claim of biological fidelity. It combines
threat and wall reflexes, food-seeking potential-field control, a tabular Q-learning residual, and
FlyGym's official hybrid locomotion controller. Training and evaluation use different randomized
arena layouts; the selected commands then drive 42 leg-joint degrees of freedom and six adhesion
channels through MuJoCo, and the combined result is written atomically:

```bash
uv run flybrain experiment practical-autonomy \
  --training-episodes 24 \
  --evaluation-episodes 12 \
  --max-steps 400 \
  --seed 7 \
  --output artifacts/practical-autonomy-v1-seed7.json \
  --video artifacts/practical-autonomy-seed7.mp4
```

The verified seed-7 run reached food in 12/12 held-out arenas, made zero threat contacts, performed
751 Q-value updates across 213 observed states, and executed 3,200 MuJoCo steps. The physical fly
used all 42 official locomotion DOFs plus six adhesion channels, moved 3.36 mm horizontally, and
finished stable at a 1.14 mm thorax height. This is an engineering demonstration with an assisted
hybrid controller, not evidence that the MaleCNS connectome alone learned full fly intelligence.

### Interactive MuJoCo application

Open the persistent real-time viewer:

```bash
uv run flybrain interactive
```

The command automatically relaunches through `mjpython` on macOS. Controls are shown inside the
window: `W/S` changes forward or reverse drive, `A/D` changes steering, `C` centers steering,
`Space` stops, `R` resets the body, `P` toggles the autonomous demonstration, and `Q` exits. The
overlay reports the current mode, forward command, turn command, and simulation time.

Verify the physical controller without opening a window:

```bash
uv run flybrain interactive --dry-run --steps 200
```

## Train and watch the autonomous game learner

Install the optional learning stack without removing FlyGym/MuJoCo:

```bash
uv sync --all-extras
```

Train the procedural one-button runner, then continue the same replay buffer and optimizer from an
atomic checkpoint:

```bash
uv run flybrain games train --game runner --visual --steps 100000 \
  --checkpoint-every 10000 --seed 7 \
  --output artifacts/games/runner/my-visual-run

uv run flybrain games train --game runner --steps 100000 --seed 7 \
  --output artifacts/games/runner/seed7-v1

uv run flybrain games train --game runner --steps 200000 --seed 7 \
  --output artifacts/games/runner/seed7-v1 \
  --resume artifacts/games/runner/seed7-v1/latest
```

Evaluate only on the immutable holdout seeds declared by the checkpoint, watch the trained agent,
or take control yourself:

```bash
uv run flybrain games evaluate --game runner \
  --checkpoint artifacts/games/runner/seed7-v1/best
uv run flybrain games watch --game runner \
  --checkpoint artifacts/games/runner/seed7-v1/best
uv run flybrain games play --game runner \
  --checkpoint artifacts/games/runner/seed7-v1/best
```

During `train --visual`, the game runs beside a graph of validation distance. Every completed
checkpoint is demonstrated on the same level for a short interval, then the next checkpoint is
shown. `P` pauses playback while training continues; `Q` closes the window and lets training
finish in the terminal. Without `--visual`, training runs without a window. The separate watch and
play windows use `Space/Up` to jump, `A` to switch human/agent, `P` to pause, `R` to reset,
`N` for the next seed, `[`/`]` for speed, and `Q` to quit. Each run stores immutable step
checkpoints, full replay state, `latest` and validation-selected `best` pointers, metrics, and
holdout reports.

The verified 100,000-step seed-7 checkpoint reached mean normalized distance `0.9668` on eight
unseen procedural levels, versus `0.1119` for seeded random play and `0.0889` for the untrained
network on the identical levels. It completed one level and reached approximately the final 4% of
the other seven; this is strong measured learning but not perfect mastery. Evaluation left the
checkpoint hash unchanged. This subsystem is engineered DQN reinforcement learning, not
MaleCNS-generated biological intelligence and not direct automation of commercial Geometry Dash.

### Connect any Gymnasium game

The universal connector accepts any local game that exposes a discrete Gymnasium action space.
It saves resumable model checkpoints and shows the real `render()` frame beside the current action,
reward, episode count, checkpoint number, and a compact activity view of the observation channels.
Use a registered environment directly:

```bash
uv run flybrain games connect \
  --env-id CartPole-v1 \
  --steps 100000 \
  --checkpoint-every 10000 \
  --visual \
  --output artifacts/games/cartpole/seed7
```

For a local game, expose a zero-argument factory in the form `module:callable`. The factory must
return a Gymnasium environment with `render_mode="rgb_array"` configured when the game supports
rendering:

```bash
uv run flybrain games connect \
  --factory my_game:make_env \
  --steps 100000 \
  --visual \
  --output artifacts/games/my-game/seed7
```

The connector validates the action space, flattens structured observations, records dependency,
space, model, and replay hashes, publishes atomic full-state checkpoints, and keeps `latest` and
validation-selected `best` pointers. Continue a run without discarding the replay buffer:

```bash
uv run flybrain games connect \
  --env-id CartPole-v1 \
  --steps 200000 \
  --resume artifacts/games/cartpole/seed7/latest \
  --output artifacts/games/cartpole/seed7
```

Evaluate the frozen policy on the checkpoint's disjoint holdout seeds; evaluation must not mutate
the checkpoint:

```bash
uv run flybrain games connect-evaluate \
  --env-id CartPole-v1 \
  --checkpoint artifacts/games/cartpole/seed7/best
```

The live window shows checkpoint playback beside training progress, reward, action, and observation
activity; it does not pretend that observation values are biological neural spikes. `P` pauses
playback and `Q` closes the window while training finishes. Continuous, structured, or pixel-heavy
action spaces need a game-specific learner adapter; the existing Chess and Runner adapters provide
those richer interfaces. This game layer is an engineered learning bridge and does not change the
evidence status of the biological MaleCNS path.

### Train, watch, or play chess

Chess uses `python-chess` for rules, a learned residual policy/value network, legal-move PUCT,
optional Stockfish teacher targets, and self-play continuation. The teacher is recorded in every
checkpoint and is not embedded in the deployed policy:

```bash
uv run flybrain games train --game chess --visual \
  --teacher-positions 128 --teacher-nodes 500 \
  --self-play-games 1 --self-play-simulations 4 \
  --max-game-plies 40 --epochs 3 --seed 7 \
  --output artifacts/games/chess/my-visual-run

uv run flybrain games train --game chess \
  --teacher-positions 128 --teacher-nodes 500 \
  --self-play-games 1 --self-play-simulations 4 \
  --max-game-plies 40 --epochs 3 --seed 7 \
  --output artifacts/games/chess/seed7-v1

uv run flybrain games evaluate --game chess \
  --checkpoint artifacts/games/chess/seed7-v1/best \
  --games-per-opponent 1 --search-simulations 4 \
  --max-game-plies 40 --stockfish-nodes 100

uv run flybrain games watch --game chess \
  --checkpoint artifacts/games/chess/seed7-v1/best
uv run flybrain games play --game chess \
  --checkpoint artifacts/games/chess/seed7-v1/best
```

Click a source and destination square to move. `F` flips the board, `N` starts a new game, `U`
undoes in human/analysis mode, `A` changes the human side, `P` pauses, `[`/`]` changes search, and
`Q` quits.

The bounded seed-7 artifact contains 128 Stockfish-500-node examples plus 32 self-play positions.
Its six short evaluation games produced zero illegal moves, four max-ply draws, and two losses.
The opponent-relative estimate was about `646` Elo with a very wide interval; it is not an official
rating, and max-ply draws do not prove equal strength. The acceptance test separately verifies that
training raises the target-move probability above `0.5`. This is a functional resumable learner,
not yet strong chess and not evidence of general or biological intelligence.

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

The hexapod command resolves 36 exact thorax-coxa/trochanter/tibia motor populations, six proprioceptive banks,
and six descending populations from one registry. It publishes direct motor calibration,
phase-gait calibration, DN-to-motor, proprio-to-motor, and stateful neural-body feedback as
separate result families; it never turns direct calibration into a sensory or locomotion claim.

```bash
uv run flybrain experiment hexapod-motor artifacts/male-cns-v1.0-w5 \
  --registry data/registry/hexapod-motor-registry-v2.json \
  --steps 180 \
  --seed 7 \
  --output artifacts/hexapod-motor-malecns-seed7.json
```

The retained run calibrates all 36 direct antagonist groups. The phase protocol now demonstrates
stable target-independent forward propulsion in the reference body, but it remains null as a
tripod-walking claim because alternating support is not established. At the 180-step causal
window, all six DN-to-motor paths are positive, four of six proprioceptive banks are positive,
and the stateful closed loop is positive under its declared thresholds. These are recorded without
turning sparse recruitment into a walking claim in
[docs/data/male-cns-hexapod-motor-v2-180.md](docs/data/male-cns-hexapod-motor-v2-180.md). The assay
is a sparse, causally tested motor foundation—not evidence of a walking or intelligent animal.

### Foreleg sensory-subtype diagnostic

The retained MaleCNS annotations can be split into foreleg proprioceptive
subclasses before applying identical per-neuron voltage pulses. This open-loop
assay records source and tibia-motor spikes, source-silencing controls, exact
replay, whole-bank/leave-one-subtype-out controls, and homologous left/right
subtype availability. Group sizes differ;
these pulses are interventions, not a calibrated body-to-sensor encoder.

```bash
uv run flybrain experiment foreleg-subtypes artifacts/male-cns-v1.0-w5 \
  --registry data/registry/hexapod-motor-registry-v2.json \
  --steps 180 --seed 7 --drive-interval-steps 25 --drive-amplitude-mv 10 \
  --output artifacts/foreleg-subtypes-local.json
```

See [the measured results](docs/data/foreleg-proprio-subtypes-2026-09-26.md).

### Experimental single-leg muscle bridge

The installed FlyGym 2.1.0 also includes FlyMimic's Hill-type muscle body. The
`flybrain.muscle_probe` assay routes four name-matched MaleCNS motor populations
to four of its 15 left-front-leg muscles; the other 11 stay at minimum control.
Published no-lesion and all-motor-lesioned MaleCNS traces produce different
muscle forces and joint angles in matched 100-ms open-loop replays. The body is
tethered, the other legs are not muscle-driven, and the cross-sex mapping is a
declared model assumption. It is **not** six-leg locomotion or learned autonomy.

```bash
FLYBRAIN_MUSCLE_ASSET_TESTS=1 .venv/bin/pytest tests/test_muscle_probe.py -q
```

See the [reuse audit and reproducible muscle result](docs/data/motor-muscle-reuse-audit-2026-09-25.md).

## Run the autonomous retained hexapod loop

The autonomous command removes the external conditioning schedule. Anonymous olfactory activity
propagates through the retained graph, physical contact recruits registered DANs, sparse local
KC-to-MBON plasticity updates existing positive edges, measured PPL101/PAM01 DANs contribute fast
dopamine and slow nitric-oxide traces, all 36 motor groups remain available to the decoder, and
six-bank proprioception returns body state to the graph. No reward scalar, target coordinate,
desired action, or hidden policy enters the controller.

Physical contact also drives only the 2,558 exact `vnc_sensory` / `mechanosensory_tactile` bodies
resolved from the biological-interface registry. Because the retained annotations do not yet map
each body-surface point to individual tactile neurons, their uniform source-event assignment is
recorded as `uniform_registered_vnc_tactile_assumption` in every episode artifact. It is a contact
reflex input, not a claim of learned threat avoidance.

```bash
uv sync --extra dev --extra physics
uv run --extra physics flybrain experiment autonomous-hexapod \
  artifacts/male-cns-v1.0-w5 \
  --steps 2 \
  --seed 7 \
  --backend reference \
  --output artifacts/autonomous-hexapod-reference-seed7.json
```

To compare annotation-aware proprioceptive source selection, set
`--proprioceptive-encoding budget_matched_uniform_spikes` or
`--proprioceptive-encoding subtype_weighted_spikes`. Both require exact retained
sensory-subtype annotations and match the number of addressed cells per leg and
neural step **when given the same body observation**. The subtype scores are
unvalidated encoding assumptions, not measured natural spike rates. The result
separately reports event objects, addressed cell occurrences, and their annotated
subtype counts; addressed cells are not necessarily observed neural spikes.
The [100-step comparison](docs/data/male-cns-autonomous-hexapod.md) did not
establish a behavioral improvement.

Use `--backend flygym` to run the same neural and torque inputs on pinned FlyGym 2.1.0 / MuJoCo
3.9.0. The retained two-step reference and FlyGym runs both preserved the graph, replayed exactly,
produced motor spikes, closed physical contact-to-DAN learning, and returned proprioception. See
[docs/data/male-cns-autonomous-hexapod.md](docs/data/male-cns-autonomous-hexapod.md) for identities,
measurements, slow-memory evidence, reproduction commands, and the remaining behavioral-validation
gates.

Run the multi-condition behavior benchmark:

```bash
uv run --extra physics flybrain experiment autonomous-behavior \
  artifacts/male-cns-v1.0-w5 \
  --training-episodes 1 \
  --holdout-episodes 3 \
  --steps 20 \
  --seed 7 \
  --seeds 7,11,13 \
  --backend reference \
  --output artifacts/autonomous-behavior-reference-seed7.json
```

The benchmark runs normal, no-plasticity, DAN-lesion, KC→MBON-lesion, and rewired controls with
separate mutable state and publishes task-specific environment-only food/threat metrics.
Current results carry `evidence_protocol="measured-replay-persistent-memory-v6"` and
per-episode evidence. The v6 rewired control swaps KC→MBON targets on an isolated
graph copy, preserves source and target edge degrees, rejects duplicate edges,
records how many edges changed, and cannot support a behavioral claim unless at
least half the declared plastic edges were actually rewired in every replicate.
Pre-v6 artifacts only shuffled plasticity-routing labels; they did **not** test a
structurally rewired graph and cannot establish superiority to that control.
Every training and evaluation episode is replayed from its pre-episode weights;
evaluation freezes plasticity. Repeated seeds are rejected. Autonomous defaults
disable the external sinusoidal leg drive; explicitly enabling it blocks the
behavioral claim. DAN silencing and KC→MBON edge removal are distinct interventions.
The v4–v6 gate fingerprints geometry and physical perturbations independently of
variant names, rejects train/holdout world overlap, and requires at least two
distinct unseen worlds for each of food and threat before a behavioral claim can
pass. The retained default evaluates four holdout worlds spanning mirrored source
geometry, shifted starts, mass/friction changes, and one delayed-motor condition.

Artifacts predating v2 (without either measured-replay v2 or v3 evidence) remain historical diagnostics:
their benchmark `replay_exact=true` was hard-coded and does **not** establish
reproducibility. Their evaluation code also allowed within-episode weight updates.
A positive narrow behavioral claim requires independent replicates, measured
replay/graph checks, frozen evaluation weights, unassisted motor decoding and
positive intervals against every declared control. Passing this benchmark alone
would not establish full fly intelligence or biological equivalence.
See [the evidence audit](docs/data/autonomy-evidence-audit-2026-09-23.md).
The measured v5 smoke artifact and exact limits are documented in
[the v5 learning/generalization audit](docs/data/autonomy-learning-generalization-v5-2026-09-23.md).
The v6 structural-control change and retained smoke evidence are documented in
[the v6 structural-control audit](docs/data/autonomy-structural-control-v6-2026-09-25.md).

The v3–v6 protocols also carry fast plasticity traces and slow dopamine/NO memory
between training episodes within each condition and seed. Holdout evaluation
freezes this complete learning state as well as its effective weights. Immutable
learning checkpoints support checked JSON round-trips and atomic file publication
through `save_learning_memory` / `load_learning_memory`. Episode results include
the resumable state; benchmark evidence records initial/final memory digests.
The physical body and fast neural voltages still reset between episodes: this
is learning-state continuation, not a checkpoint of the entire embodied agent.
See [memory continuation](docs/data/learning-memory-continuation-2026-09-23.md).

An earlier long-horizon diagnostic includes passive joint-restoring mechanics. It prevents
the reference body from collapsing over six seconds, but the 3-seed/2-holdout/100-step artifact
`artifacts/autonomous-behavior-reference-seeds7-11-13-train1-holdout2-steps100-passive-elasticity.json`
still has `behavioral_claim_allowed=false`: every food and threat interval crosses zero. The next
investigation includes bilateral olfactory evidence for steering. Arthropod navigation relies on comparing sensor signals and
integrating them over time ([Steele, Lanz & Nagel, 2023](https://doi.org/10.1007/s00359-022-01611-9)).
### External biological boundary for odor navigation

The autonomous hexapod assay currently uses measured bilateral ORN IDs, but its
arena odor field is still a model assumption: a distance-based concentration field
with deterministic sinusoidal temporal modulation. This must not be described as a reconstructed fly plume.
That distinction matters because arthropod odor navigation uses concentration
comparisons integrated over time, while recent Drosophila work also implicates
odor-motion cues, visual reafference, and compass-related descending control.

External checks used for the design boundary:

- Steele, Lanz & Nagel, *Olfactory navigation in arthropods*,
  https://doi.org/10.1007/s00359-022-01611-9
- Rayshubskiy et al., *DNa01/DNa02 steering*,
  https://doi.org/10.7554/eLife.102230
- Bates et al., *Distributed brain-and-cord control*,
  https://doi.org/10.1038/s41586-026-10735-w

Therefore `behavioral_claim_allowed` remains false until a causal benchmark
with a temporally structured odor field, all declared controls, multiple seeds,
and positive food and threat confidence intervals passes.
