from pathlib import Path

import pytest

from flybrain.games.artifacts import SeedSchedule, write_json_atomic


def test_seed_schedule_is_reproducible_and_disjoint() -> None:
    first = SeedSchedule.build(7, train_count=8, validation_count=4, holdout_count=4)
    second = SeedSchedule.build(7, train_count=8, validation_count=4, holdout_count=4)

    assert first == second
    assert not (set(first.training) & set(first.validation))
    assert not (set(first.training) & set(first.holdout))
    assert not (set(first.validation) & set(first.holdout))


def test_atomic_json_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_json_atomic(output, {"ok": True})

    with pytest.raises(FileExistsError):
        write_json_atomic(output, {"ok": False})

    assert output.read_text() == '{"ok":true}\n'
