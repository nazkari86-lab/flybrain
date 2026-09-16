# Biological interface registry and causal steering design

## Goal

Replace the embodied smoke test's arbitrary sensory and motor neuron IDs with snapshot-bound,
literature-backed MaleCNS populations, then establish the first causal steering and retreat assay.
The phase must make every biological claim traceable to either the canonical annotation table or a
named experimental source. It must keep biological evidence, simulator assumptions, and measured
simulation outcomes separate.

This is the next gate toward learned food seeking and threat avoidance. It does not claim a complete
retina, realistic six-leg gait, general intelligence, or successful sensory-to-behavior learning.

## Approaches considered

### Evidence-bound descending-neuron interface (chosen)

Resolve sensory and descending-neuron populations from canonical annotations, bind each behavioral
role to published evidence, and test side-specific DNa02/DNg13 steering plus MDN retreat under
controlled stimulation, silencing, reversal, and replay. This gives the strongest causal and
provenance guarantees without introducing a hidden policy.

### Direct muscle motor map

Decode specific leg motor neurons and muscle groups. This is anatomically more detailed, but the
current planar body has no joint phases, muscle dynamics, or six-leg biomechanics. Applying this
mapping now would imply precision the body model cannot express. It is the intended next motor
stage after the descending interface is validated.

### Learned decoder or deep reinforcement learning

Fit a neural decoder from connectome activity to successful actions. This could improve task score
quickly, but it would let an unmeasured artificial policy compensate for incorrect biology. It is
allowed only as an explicitly separated baseline and is forbidden in the primary agent.

## Evidence boundary

The registry assigns every assertion one of four kinds:

1. `dataset_measurement`: an exact value resolved from the retained MaleCNS snapshot;
2. `experimental_result`: a claim stated by a cited biological experiment;
3. `model_assumption`: a simulator choice that is not presented as measured biology;
4. `simulation_observation`: a value produced by a versioned run artifact.

The implementation and reports must never promote a model assumption or simulation observation to
an experimental result. A null network response remains a valid observation and must not be hidden
by retuning the acceptance threshold after the run.

The first registry version cites at least these sources:

- Yang et al., *Cell* (2024), PMID 39293446, DOI 10.1016/j.cell.2024.08.033:
  left/right DNa02 and DNg13 activity predicts ipsiversive steering; unilateral activation drives
  ipsiversive rotation; DNa02 shortens ipsilateral strides and DNg13 lengthens contralateral strides.
- Bidaye et al. (2017), PMID 28238656: MDN activity is necessary and sufficient for backward
  walking, and asymmetric MDN activation contributes to directional retreat from looming input.
- Fujiwara et al. (2018), PMID 30528583: horizontal optic-flow pathways affect turning in opposite
  directions and bilateral signals affect walking speed.
- The retained MaleCNS `source-annotations.parquet` and its snapshot identity for all neuron IDs,
  classes, sides, and population counts.

PMID 40690376 may be recorded as corroborating steering evidence, but no behavior is assigned from
its abstract alone unless the exact cell type and causal sign are established in the cited record.

## Canonical populations

Population resolution uses exact equality predicates over canonical columns. Results are sorted by
`bodyId`, deduplicated, and required to be non-empty. IDs are never selected by row order, numeric
range, graph degree, or a convenient fixed count.

The first retained snapshot resolves the following descending populations:

| Role | Selector | Expected IDs |
| --- | --- | --- |
| left fine steering | `superclass=descending_neuron`, `type=DNa02`, `somaSide=L` | `523769` |
| right fine steering | same, `somaSide=R` | `10360` |
| left broad steering | `type=DNg13`, `somaSide=L` | `11074` |
| right broad steering | `type=DNg13`, `somaSide=R` | `512006` |
| left retreat | `type=MDN`, `somaSide=L` | `11288`, `12348` |
| right retreat | `type=MDN`, `somaSide=R` | `10763`, `11332` |

The `superclass=ol_sensory`, `class=visual` selector resolves 6,091 visual sensory neurons in the
retained snapshot. Left and right input banks are separated by `rootSide`. The source table has no
non-null `assignedOlHex1` or `assignedOlHex2` values for these neurons, so this phase must not claim
an ommatidial or retinotopic map. R1-R6 neurons are used for luminance/motion-energy input; R7/R8
types remain available for later spectral assays but do not receive invented color semantics.

Other declared populations are resolved for future interfaces but are not evidence of a working
behavior: central-brain olfactory sensory neurons use `superclass=cb_sensory`, `class=olfactory`;
VNC tactile neurons use `superclass=vnc_sensory`, `class=mechanosensory_tactile`; and VNC
proprioceptors use `superclass=vnc_sensory`, `class=mechanosensory_proprioceptive`.

## Registry architecture

`BiologicalInterfaceRegistry` loads a versioned declarative registry and resolves it against one
annotation table. An `AnnotationSelector` supports only an allowlisted set of columns and exact
`equals`/`in` predicates. A `ResolvedPopulation` stores the ordered IDs, selector, count, annotation
file SHA-256, canonical snapshot identity, and an ID digest. An `EvidenceRecord` stores source URL,
PMID or DOI where applicable, the exact supported claim, evidence kind, and confidence.

Resolution fails before simulation when:

- a requested column is absent or a selector is empty;
- a population has duplicate IDs or unexpected laterality;
- an optional expected-ID or expected-count assertion differs from the retained snapshot;
- an evidence record lacks a source or tries to mark a simulator assumption as biology;
- the registry, annotation file, and graph do not share the same snapshot identity;
- a declared sensory or output ID is absent from the executed graph.

The resolver emits a JSON-safe manifest. Runs embed this manifest rather than only the final ID
lists, so later annotation changes cannot silently alter an experiment.

## Sensory assay

The benchmark adds an observational `RetinalObservation` produced from the planar arena. It contains
bounded left/right horizontal motion energy and bounded bilateral angular-expansion energy. These
values are computed from consecutive body-relative angular projections; the observation exposes no
food label, threat label, target coordinate, desired heading, or action hint.

Motion energy drives only the corresponding left/right R1-R6 banks using non-negative event rates
or voltages. Opposite image motion swaps the banks. Looming is represented by increasing bilateral
angular expansion, not by a hard-coded `threat=true` signal. Because the retained MaleCNS table does
not provide ommatidial coordinates, this stage is explicitly a hemispheric optic-flow interface,
not a full compound-eye model.

Three protocols remain separate:

1. `dn_calibration` directly stimulates one declared DN population to validate decoder sign and
   causal intervention machinery;
2. `sensory_open_loop` supplies mirrored optic-flow or looming sequences and measures whether the
   connectome recruits the declared DNs;
3. `sensory_closed_loop` lets decoded DN spikes rotate or reverse the body and feeds the resulting
   retinal observation into the next neural chunk.

Direct DN calibration is never reported as spontaneous visual steering.

## Descending-neuron decoder

The decoder reads only spikes from registry-resolved DNa02, DNg13, and MDN populations. It does not
read world state, sensory values, target coordinates, reward, or non-neural labels.

For each side, DNa02 and DNg13 rates are normalized independently by population size and a declared
time window. Left-cell activity contributes an ipsiversive left command and right-cell activity an
ipsiversive right command, matching the cited unilateral perturbation evidence. DNa02 and DNg13
remain separate trace channels; a versioned model assumption combines them into bounded yaw only
at the body interface. Equal bilateral steering activity yields zero yaw.

MDN activity contributes bounded reverse thrust. Equal bilateral MDN activity produces straight
retreat; side imbalance adds a separately reported retreat-turn component. DNa02/DNg13 are not used
as ordinary forward-speed controls because the cited experiments do not support that interpretation.
The assay therefore starts with a declared, target-independent walking state and reports it as a
model assumption. A future gait controller must replace this assay drive before realistic walking
is claimed.

The decoder supports explicit masks by population and side. Silencing changes only selected neural
outputs; it must not substitute an opposite motor command.

## Causal benchmark

Each stimulus family is run with the same graph, initial state, seed, parameters, and duration in
the following conditions:

- normal;
- mirrored stimulus;
- left-population silence;
- right-population silence;
- bilateral silence;
- restoration after silence;
- identical-seed replay;
- bounded timing and amplitude perturbations;
- holdout arena geometry not used to choose parameters.

The benchmark records stimulus traces, injected events, all relevant DN spike rasters, decoded
commands, body trajectories, intervention masks, graph digests, population manifests, and resource
use. It reports effect sizes and paired differences, not only a Boolean score.

The phase passes its infrastructure and causal-calibration gate only if:

1. registry resolution exactly matches the retained IDs and identities;
2. unilateral DNa02 and DNg13 calibration produces the cited ipsiversive decoder sign, mirrored
   activation reverses it, bilateral symmetry cancels yaw, and the appropriate silence removes it;
3. bilateral MDN calibration produces retreat, side reversal mirrors the retreat-turn component,
   and MDN silence removes retreat;
4. restoration reproduces the unsilenced trace, identical replay is exact, and graph topology plus
   canonical weights remain unchanged;
5. optic-flow reversal swaps sensory banks and looming amplitude increases monotonically with
   projected angular expansion;
6. registry or provenance mismatch fails closed without publishing a partial result.

The real sensory-to-DN and closed-loop outcomes are hypothesis tests, not values the implementation
may force to pass. Their report must classify each family as positive, null, directionally wrong,
or underpowered. A positive behavioral claim additionally requires the expected direction in the
normal condition, loss or reversal under the matching lesion, restoration, exact replay, and the
same effect direction in holdout arenas. Failure of that stronger gate does not invalidate the
instrument; it prevents a steering claim and defines the next scientific repair.

## Testing and publication

Fixture tests cover selector semantics, stable ordering, expected-ID drift, missing annotations,
evidence validation, side-aware encoding, decoder cancellation, every lesion, restoration,
deterministic perturbations, graph immutability, and atomic output. Synthetic circuits provide
known positive and null sensory-to-DN cases without weakening the real-data criteria.

An opt-in real-data test resolves the complete retained annotation table and executes the selected
protocols on the full sparse MaleCNS graph. Dense neuron-by-neuron matrices are forbidden. The CLI
publishes one atomic JSON artifact containing all identities, evidence records, resolved selectors,
protocol parameters, traces or trace digests, causal comparisons, and claim classifications. It
refuses overwrite and publishes nothing on validation or execution failure.

## Next gates

After this phase, the project proceeds in this order:

1. direct motor-neuron and muscle mapping with a phase-aware six-leg body;
2. learned food seeking and threat avoidance using mushroom-body plasticity, causal lesions, and
   holdout worlds;
3. richer compound-eye geometry and receptor-aware neural dynamics where measured parameters exist;
4. FlyGym or equivalent whole-body integration and a continual-learning curriculum;
5. game benchmarks only after the same no-hidden-policy and causal-evidence gates pass.

