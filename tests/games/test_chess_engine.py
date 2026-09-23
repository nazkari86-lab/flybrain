from pathlib import Path

from flybrain.games.chess_engine import resolve_stockfish


def test_explicit_missing_stockfish_returns_none(tmp_path: Path) -> None:
    assert resolve_stockfish(tmp_path / "missing") is None


def test_path_resolution_never_invents_engine_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    assert resolve_stockfish(None) is None
