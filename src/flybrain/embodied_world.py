"""Deterministic bounded planar world and fly body state."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class ArenaConfig:
    """Numerical limits for the planar world."""

    width: float
    height: float
    dt_s: float
    wall_restitution: float

    def __post_init__(self) -> None:
        for name in ("width", "height", "dt_s", "wall_restitution"):
            _finite(getattr(self, name), name)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if not 0 <= self.wall_restitution <= 1:
            raise ValueError("wall_restitution must be between 0 and 1")


@dataclass(frozen=True)
class FlyBody:
    """Reduced physical state exposed to the sensory interface."""

    x: float
    y: float
    heading_rad: float
    forward_speed: float
    angular_speed: float
    energy: float
    leg_contacts: tuple[bool, ...]

    def __post_init__(self) -> None:
        for name in ("x", "y", "heading_rad", "forward_speed", "angular_speed", "energy"):
            _finite(getattr(self, name), name)
        if len(self.leg_contacts) != 6:
            raise ValueError("leg_contacts must contain six legs")
        if self.energy < 0:
            raise ValueError("energy must be non-negative")


@dataclass(frozen=True)
class MotorCommand:
    """Normalized motor output, with no target-directed semantics."""

    forward: float
    turn: float

    def __post_init__(self) -> None:
        for name in ("forward", "turn"):
            value = _finite(getattr(self, name), name)
            if not -1 <= value <= 1:
                raise ValueError(f"{name} must be between -1 and 1")


@dataclass(frozen=True)
class WorldStep:
    """Result of integrating one body step."""

    body: FlyBody
    food_contact: bool
    threat_contact: bool
    wall_contact: bool


def _distance_sq(first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


class ArenaWorld:
    """A small deterministic arena with circular food and threat objects."""

    contact_radius = 0.35
    max_speed = 3.0
    max_angular_speed = 8.0

    def __init__(
        self,
        config: ArenaConfig,
        body: FlyBody,
        *,
        food: tuple[float, float],
        threat: tuple[float, float],
        wall_segments: tuple[tuple[float, float, float, float], ...] = (),
    ) -> None:
        self.config = config
        self.body = body
        self.food = self._point(food, "food")
        self.threat = self._point(threat, "threat")
        self.wall_segments = tuple(wall_segments)
        self._validate_body(body)
        for segment in self.wall_segments:
            if len(segment) != 4 or not all(math.isfinite(value) for value in segment):
                raise ValueError("wall segments must contain four finite coordinates")

    @staticmethod
    def _point(point: tuple[float, float], name: str) -> tuple[float, float]:
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError(f"{name} must contain two finite coordinates")
        return float(point[0]), float(point[1])

    def _validate_body(self, body: FlyBody) -> None:
        if not 0 <= body.x <= self.config.width or not 0 <= body.y <= self.config.height:
            raise ValueError("body must start inside arena bounds")

    def step(self, command: MotorCommand) -> WorldStep:
        dt = self.config.dt_s
        speed = max(
            -self.max_speed,
            min(self.max_speed, self.body.forward_speed + command.forward * 3.0 * dt),
        )
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.body.angular_speed + command.turn * 10.0 * dt),
        )
        heading = (self.body.heading_rad + angular * dt + math.pi) % (2 * math.pi) - math.pi
        x = self.body.x + speed * math.cos(heading) * dt
        y = self.body.y + speed * math.sin(heading) * dt
        wall_contact = False
        if x < 0 or x > self.config.width:
            x = max(0.0, min(self.config.width, x))
            speed = -speed * self.config.wall_restitution
            wall_contact = True
        if y < 0 or y > self.config.height:
            y = max(0.0, min(self.config.height, y))
            speed = -speed * self.config.wall_restitution
            wall_contact = True
        next_body = FlyBody(
            x=x,
            y=y,
            heading_rad=heading,
            forward_speed=speed,
            angular_speed=angular,
            energy=max(
                0.0,
                self.body.energy
                - dt * (0.002 * abs(command.forward) + 0.001 * abs(command.turn)),
            ),
            leg_contacts=(wall_contact,) * 6,
        )
        if self.wall_segments:
            wall_contact = wall_contact or self._segment_contact(next_body.x, next_body.y)
        self.body = next_body
        position = (next_body.x, next_body.y)
        return WorldStep(
            body=next_body,
            food_contact=_distance_sq(position, self.food) <= self.contact_radius**2,
            threat_contact=_distance_sq(position, self.threat) <= self.contact_radius**2,
            wall_contact=wall_contact,
        )

    def _segment_contact(self, x: float, y: float) -> bool:
        for x1, y1, x2, y2 in self.wall_segments:
            if x1 == x2 and min(y1, y2) <= y <= max(y1, y2) and abs(x - x1) <= self.contact_radius:
                return True
            if y1 == y2 and min(x1, x2) <= x <= max(x1, x2) and abs(y - y1) <= self.contact_radius:
                return True
        return False
