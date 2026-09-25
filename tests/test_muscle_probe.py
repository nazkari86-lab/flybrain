import math
import os

import pytest

from flybrain.flygym_backend import flygym_availability
from flybrain.hexapod_motor import CANONICAL_MOTOR_GROUPS

MUSCLE_NAMES = (
    "LFC_tergopleural_promotor_a",
    "LFC_tergopleural_promotor_b",
    "LFC_pleural_remotor_and_abductor",
    "LFC_pleural_promotor",
    "LFC_sternal_anterior_rotator",
    "LFC_sternal_posterior_rotator",
    "LFC_sternal_adductor",
    "LFF_trochanter_flexor_b",
    "LFF_sterno-tergo-trochanter_extensor_a",
    "LFF_sterno-tergo-trochanter_extensor_b",
    "LFF_accesory_trochanter_flexor",
    "LFF_trochanter_extensor",
    "LFF_trochanter_flexor_a",
    "LFTibia_flex_93434",
    "LFTibia_extensor_93932",
)


def test_muscle_action_routes_only_four_exact_left_fore_populations() -> None:
    from flybrain.muscle_probe import muscle_action

    names = [item[0] for item in CANONICAL_MOTOR_GROUPS]
    values = [0.0] * len(names)
    values[names.index("left_fore_thorax_coxa_anterior")] = 0.2
    values[names.index("left_fore_thorax_coxa_posterior")] = 0.4
    values[names.index("left_fore_tibia_flexor")] = 0.6
    values[names.index("left_fore_tibia_extensor")] = 0.8
    values[names.index("right_fore_tibia_extensor")] = 1.0

    action = muscle_action(values, MUSCLE_NAMES)

    assert len(action) == 15
    assert action[MUSCLE_NAMES.index("LFC_sternal_anterior_rotator")] == pytest.approx(0.2)
    assert action[MUSCLE_NAMES.index("LFC_sternal_posterior_rotator")] == pytest.approx(0.4)
    assert action[MUSCLE_NAMES.index("LFTibia_flex_93434")] == pytest.approx(0.6)
    assert action[MUSCLE_NAMES.index("LFTibia_extensor_93932")] == pytest.approx(0.8)
    assert all(
        action[i] == pytest.approx(0.0001)
        for i, name in enumerate(MUSCLE_NAMES)
        if name not in {
            "LFC_sternal_anterior_rotator",
            "LFC_sternal_posterior_rotator",
            "LFTibia_flex_93434",
            "LFTibia_extensor_93932",
        }
    )


def test_muscle_action_fails_closed_on_missing_target_or_bad_activations() -> None:
    from flybrain.muscle_probe import muscle_action

    zeros = [0.0] * len(CANONICAL_MOTOR_GROUPS)
    with pytest.raises(ValueError, match="missing muscle"):
        muscle_action(zeros, MUSCLE_NAMES[:-1])
    with pytest.raises(ValueError, match="36"):
        muscle_action(zeros[:-1], MUSCLE_NAMES)
    bad = zeros.copy()
    bad[0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        muscle_action(bad, MUSCLE_NAMES)
    bad[0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        muscle_action(bad, MUSCLE_NAMES)


@pytest.mark.skipif(
    not flygym_availability().available or os.environ.get("FLYBRAIN_MUSCLE_ASSET_TESTS") != "1",
    reason="FlyGym 2.1.0 and opt-in FlyMimic assets required",
)
def test_tethered_muscle_probe_reacts_to_tibia_motor_input() -> None:
    from flybrain.muscle_probe import replay_muscle_trace

    group_names = [item[0] for item in CANONICAL_MOTOR_GROUPS]
    active = [0.0] * len(group_names)
    active[group_names.index("left_fore_tibia_extensor")] = 0.8
    silent = [0.0] * len(group_names)

    driven = replay_muscle_trace([active] * 30)
    control = replay_muscle_trace([silent] * 30)

    muscle = "LFTibia_extensor_93932"
    joint = "joint_LFTibia_pitch"
    assert driven["steps"] == control["steps"] == 30
    assert driven["muscle_names"] == control["muscle_names"] == MUSCLE_NAMES
    assert driven["mean_abs_muscle_force"][muscle] > (
        control["mean_abs_muscle_force"][muscle] + 1.0
    )
    joint_delta = driven["final_joint_angles_rad"][joint] - control["final_joint_angles_rad"][joint]
    assert abs(joint_delta) > 0.01
    assert driven["closed_loop"] is False
