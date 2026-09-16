"""World-to-spike and spike-to-motor interfaces for embodied episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flybrain.embodied_world import MotorCommand, WorldStep


def _ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique IDs")
    return values


@dataclass(frozen=True)
class SensoryMap:
    """Declared neuron populations used by the sensory encoder."""

    visual_ids: tuple[int, ...]
    odor_ids: tuple[int, ...]
    touch_ids: tuple[int, ...]
    proprioception_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (
            _ids(self.visual_ids, "visual_ids"),
            _ids(self.odor_ids, "odor_ids"),
            _ids(self.touch_ids, "touch_ids"),
            _ids(self.proprioception_ids, "proprioception_ids"),
        )
        flattened = [value for group in groups for value in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("sensory populations must be disjoint")


@dataclass(frozen=True)
class ExternalEvent:
    """A bounded source-equivalent voltage event for the Shiu simulator."""

    step: int
    neuron_ids: tuple[int, ...]
    voltages: tuple[float, ...]
    channel: str

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.neuron_ids or len(self.neuron_ids) != len(self.voltages):
            raise ValueError("event IDs and voltages must be non-empty and parallel")
        if any(value <= 0 for value in self.neuron_ids):
            raise ValueError("event neuron IDs must be positive")
        if not self.channel:
            raise ValueError("channel must not be empty")
        if not all(math.isfinite(value) and value >= 0 for value in self.voltages):
            raise ValueError("event voltages must be finite and non-negative")


class SensoryEncoder:
    """Encode observable world state into deterministic quantized voltage events."""

    version = "embodied-sensors-v1"

    def __init__(
        self,
        mapping: SensoryMap,
        *,
        arena_width: float,
        arena_height: float,
        voltage_scale: float = 68.75,
    ) -> None:
        if not math.isfinite(arena_width) or not math.isfinite(arena_height):
            raise ValueError("arena dimensions must be finite")
        if arena_width <= 0 or arena_height <= 0:
            raise ValueError("arena dimensions must be positive")
        if not math.isfinite(voltage_scale) or voltage_scale <= 0:
            raise ValueError("voltage_scale must be finite and positive")
        self.mapping = mapping
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.voltage_scale = voltage_scale

    def encode(self, world_step: WorldStep, *, step: int) -> tuple[ExternalEvent, ...]:
        body = world_step.body
        diagonal = math.hypot(self.arena_width, self.arena_height)
        visual = max(0.0, min(1.0, 1.0 - math.hypot(body.x, body.y) / diagonal))
        food_distance = math.hypot(
            body.x - world_step.food_position[0], body.y - world_step.food_position[1]
        )
        odor = 1.0 if world_step.food_contact else max(0.0, min(1.0, 1.0 / (1.0 + food_distance)))
        touch = (
            1.0
            if world_step.food_contact or world_step.threat_contact or world_step.wall_contact
            else 0.0
        )
        proprioception = max(0.0, min(1.0, abs(body.forward_speed) / 3.0))
        values = (
            ("visual", self.mapping.visual_ids, visual),
            ("odor", self.mapping.odor_ids, odor),
            ("touch", self.mapping.touch_ids, touch),
            ("proprioception", self.mapping.proprioception_ids, proprioception),
        )
        return tuple(
            ExternalEvent(
                step,
                ids,
                tuple(self.voltage_scale * value for _ in ids),
                channel,
            )
            for channel, ids, value in values
            if ids
        )


@dataclass(frozen=True)
class MotorMap:
    """Declared output populations and their fixed decoder semantics."""

    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    forward_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (
            _ids(self.left_ids, "left_ids"),
            _ids(self.right_ids, "right_ids"),
            _ids(self.forward_ids, "forward_ids"),
        )
        flattened = [value for group in groups for value in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("motor populations must be disjoint")


class MotorDecoder:
    """Convert output spike counts to bounded forward/turn commands."""

    version = "motor-v1"

    def __init__(self, mapping: MotorMap) -> None:
        self.mapping = mapping
        self._left = set(mapping.left_ids)
        self._right = set(mapping.right_ids)
        self._forward = set(mapping.forward_ids)

    def decode(self, spikes: tuple[int, ...]) -> MotorCommand:
        left = sum(neuron_id in self._left for neuron_id in spikes)
        right = sum(neuron_id in self._right for neuron_id in spikes)
        forward = sum(neuron_id in self._forward for neuron_id in spikes)
        turn = (left - right) / max(1, max(left, right))
        thrust = forward / max(1, len(self.mapping.forward_ids))
        return MotorCommand(
            forward=max(-1.0, min(1.0, thrust)),
            turn=max(-1.0, min(1.0, turn)),
        )
