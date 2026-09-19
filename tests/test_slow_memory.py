from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flybrain.slow_memory import (
    SlowMemoryParameters,
    SlowMemoryState,
    resolve_nitric_oxide_dans,
)


def test_no_dan_resolver_selects_only_measured_ppl101_and_pam01(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(
        pa.table(
            {
                "bodyId": [1, 2, 3, 4],
                "class": ["DAN", "DAN", "DAN", "MBON"],
                "type": ["PPL101", "PAM01", "PAM02", "MBON26"],
            }
        ),
        snapshot / "source-annotations.parquet",
    )

    resolved = resolve_nitric_oxide_dans(snapshot)

    assert resolved.ppl101_ids == (1,)
    assert resolved.pam01_ids == (2,)
    assert resolved.all_ids == (1, 2)
    assert resolved.evidence_doi == "10.7554/eLife.49257"


def test_measured_slow_da_no_dynamics_are_multiplicative_and_no_specific() -> None:
    params = SlowMemoryParameters()
    state = SlowMemoryState.initial(
        nitric_oxide_competent=np.array([True, False], dtype=np.bool_)
    )

    state.step(
        paired=np.array([True, True], dtype=np.bool_),
        dan_unpaired=np.array([False, False], dtype=np.bool_),
        dt_seconds=60.0,
        parameters=params,
    )

    assert state.dopamine_immediate.tolist() == pytest.approx([1 - np.exp(-4.3)] * 2)
    assert state.nitric_oxide_immediate[0] == pytest.approx(1 - np.exp(-0.96))
    assert state.nitric_oxide_immediate[1] == 0.0
    assert state.weight_multipliers[0] == pytest.approx(
        (1 - state.dopamine_effect[0]) * (1 + state.nitric_oxide_effect[0])
    )
    assert state.weight_multipliers[1] == pytest.approx(1 - state.dopamine_effect[1])


def test_unpaired_dan_activation_recovers_immediate_memory_toward_baseline() -> None:
    state = SlowMemoryState.initial(
        nitric_oxide_competent=np.array([True], dtype=np.bool_)
    )
    params = SlowMemoryParameters()
    state.step(
        paired=np.array([True], dtype=np.bool_),
        dan_unpaired=np.array([False], dtype=np.bool_),
        dt_seconds=60.0,
        parameters=params,
    )
    before = (state.dopamine_immediate[0], state.nitric_oxide_immediate[0])

    state.step(
        paired=np.array([False], dtype=np.bool_),
        dan_unpaired=np.array([True], dtype=np.bool_),
        dt_seconds=60.0,
        parameters=params,
    )

    assert state.dopamine_immediate[0] < before[0]
    assert state.nitric_oxide_immediate[0] < before[1]
