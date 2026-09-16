# Task 2 report: reviewed MaleCNS registry artifact

## Status

Implemented, verified, and ready for task review. The delegated worker hit an
external usage limit after adding the planned RED test; the controller then
completed the bounded Task 2 artifact work in this same worktree without
changing Task 1 production code.

## Implementation

- Added `data/registry/biological-interface-registry-v1.json`.
- Registered exact DNa02, DNg13, MDN, and HS IDs measured from the retained
  source annotations.
- Registered expected counts for visual, R1-R6 hemispheric, LC16 hemispheric,
  olfactory, tactile, and proprioceptive populations.
- Bound every population to the official MaleCNS source or the cited Cell,
  MDN-retreat, and optic-flow literature records.
- Added the exact expected-population assertions to
  `tests/test_biological_registry.py`.

## TDD evidence

The delegated worker created the failing test before the JSON existed. After
the JSON was added, the focused test became green:

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest tests/test_biological_registry.py -q
40 passed in 0.55s
```

Full retained-snapshot resolution:

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/python -c 'from pathlib import Path; from flybrain.biological_registry import load_biological_registry, resolve_biological_registry; r=load_biological_registry(Path("data/registry/biological-interface-registry-v1.json")); x=resolve_biological_registry(r, Path("/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5")); print([(p.name, len(p.neuron_ids)) for p in x.populations])'
```

```text
[("visual_all", 6091), ("visual_r1_r6_left", 1112), ("visual_r1_r6_right", 2265), ("hs_left", 4), ("hs_right", 4), ("lc16_left", 88), ("lc16_right", 94), ("olfactory_cb", 2639), ("tactile_vnc", 2558), ("proprioceptive_vnc", 1030), ("d_na02_left", 1), ("d_na02_right", 1), ("d_ng13_left", 1), ("d_ng13_right", 1), ("mdn_left", 2), ("mdn_right", 2)]
```

## Quality gates

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/ruff check src/flybrain/biological_registry.py tests/test_biological_registry.py
All checks passed!

PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/mypy src/flybrain/biological_registry.py
Success: no issues found in 1 source file

PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest -q
144 passed, 3 skipped in 1.02s
```

## Self-review

- Every exact ID and count resolves against the retained annotation Parquet;
  no row-order or truncation selection is present.
- HS selectors use `in_values` for the four measured cell types and exact
  side-specific IDs.
- LC16 selectors assert the measured left/right counts and retain every
  resolved ID in the later artifact manifest.
- Experimental records have valid PMID/DOI evidence under Task 1 strict
  validation; the official dataset record remains a dataset measurement.
- No production code, graph topology, or ignored generated artifact was
  modified by this task.

## Concerns and boundaries

- This registry proves identity and provenance, not sensory-to-DN behavior or
  intelligence. Those claims belong to later benchmark tasks.
- The real snapshot is outside this worktree by design and was read from its
  absolute retained path; the registry JSON is the versioned source declaration.

## Fix round 1/5

### Status

Implemented and independently verified in this worktree. The fixes address all
opened review findings for the Task 2 artifact.

### Changes

- Added four explicit `model_assumption` records for R1-R6 hemispheric mapping,
  decoder sign/normalization, HS/LC16 calibration, and the upstream visual
  processing bypass; each is bound to the relevant population declarations.
- Renamed the optic-flow record to `busch-2018-optic-flow` and corrected the
  claim to match PMID 30528583 attribution.
- Added a committed test that invokes `resolve_biological_registry` against the
  exact retained path `/Users/dulatnurlanuly/Downloads/flybrain/artifacts/male-cns-v1.0-w5`.
  It skips only when that exact directory is unavailable and asserts population
  names, snapshot dataset identity, manifest identity, all exact DN/MDN/HS IDs,
  and all declared counts.
- Extended `EvidenceRecord` with machine-checkable source artifact name, URL,
  and SHA-256 fields. The official record is bound to the exact MaleCNS v1.0
  body-annotations URL and checksum from `data/registry/sources.yaml`.
- Added exact LC16 left/right IDs and asserted the complete v1 population-name
  set.

### TDD and verification evidence

The new evidence/binding test was run RED before the registry fixes:

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest tests/test_biological_registry.py -q
1 failed, 41 passed in 0.62s
```

After the fixes:

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest tests/test_biological_registry.py -q
42 passed in 0.36s
```

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/ruff check src/flybrain/biological_registry.py tests/test_biological_registry.py
All checks passed!
```

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/mypy src
Success: no issues found in 26 source files
```

```text
PYTHONPATH=src /Users/dulatnurlanuly/Downloads/flybrain/.venv/bin/pytest -q
146 passed, 3 skipped in 1.06s
```

`git diff --check` also completed with no output and exit status 0.

### Fix-round concerns

- The retained snapshot is an external local artifact and is intentionally not
  copied into this Git worktree; the committed test fails rather than silently
  substituting a fixture when the exact path exists but does not resolve.
- The registry remains an evidence-bound interface declaration. It does not yet
  prove steering, retreat, intelligence, or learned behavior; those require the
  later causal closed-loop benchmark.
