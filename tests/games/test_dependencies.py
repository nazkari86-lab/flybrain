from importlib.util import find_spec


def test_game_dependencies_are_importable() -> None:
    for module in ("gymnasium", "stable_baselines3", "torch", "pygame"):
        assert find_spec(module) is not None, module
