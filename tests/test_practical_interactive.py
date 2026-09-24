import json

from typer.testing import CliRunner

from flybrain.cli import app
from flybrain.practical_interactive import InteractiveControls

runner = CliRunner()


def test_interactive_controls_apply_incremental_commands_and_mode_toggles() -> None:
    controls = InteractiveControls()

    controls.handle_key(ord("W"))
    controls.handle_key(ord("W"))
    controls.handle_key(ord("A"))

    assert controls.forward == 0.4
    assert controls.turn == 0.2
    assert controls.command(0.0).forward == 0.4
    assert controls.command(0.0).turn == 0.2

    controls.handle_key(ord(" "))
    assert controls.command(0.0).forward == 0.0
    assert controls.command(0.0).turn == 0.0

    controls.handle_key(ord("P"))
    assert controls.autopilot is True
    assert controls.command(1.0).forward > 0.0

    controls.handle_key(ord("R"))
    assert controls.consume_reset_request() is True
    assert controls.consume_reset_request() is False

    controls.handle_key(ord("Q"))
    assert controls.quit_requested is True


def test_interactive_cli_headless_smoke_uses_complete_physical_controller() -> None:
    result = runner.invoke(app, ["interactive", "--dry-run", "--steps", "100"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["controlled_joint_dofs"] == 42
    assert payload["adhesion_channels"] == 6
    assert payload["steps"] == 100
    assert payload["all_actions_finite"] is True
