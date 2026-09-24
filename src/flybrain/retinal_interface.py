"""Label-free hemispheric retinal observations and visual feature calibrations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flybrain.embodied_interfaces import ExternalEvent
from flybrain.embodied_world import FlyBody


def _bounded(value: float, name: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return float(value)


@dataclass(frozen=True)
class VisualDisc:
    """Anonymous circular visual object used only by the geometric sensor."""

    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.radius)):
            raise ValueError("visual disc geometry must be finite")
        if self.radius <= 0:
            raise ValueError("visual disc radius must be positive")


@dataclass(frozen=True)
class RetinalObservation:
    """Bounded anonymous hemispheric visual features, without object labels."""

    left_luminance: float
    right_luminance: float
    left_contrast: float
    right_contrast: float
    left_motion: float
    right_motion: float
    looming: float

    def __post_init__(self) -> None:
        for name in (
            "left_luminance",
            "right_luminance",
            "left_contrast",
            "right_contrast",
            "left_motion",
            "right_motion",
            "looming",
        ):
            _bounded(getattr(self, name), name)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _angular_delta(current: float, previous: float) -> float:
    raw = current - previous
    return _wrap_angle(raw)


def _disc_features(body: FlyBody, disc: VisualDisc) -> tuple[float, float]:
    dx = disc.x - body.x
    dy = disc.y - body.y
    distance = math.hypot(dx, dy)
    if distance <= disc.radius:
        raise ValueError("visual disc may not overlap the body")
    bearing = _wrap_angle(math.atan2(dy, dx) - body.heading_rad)
    half_width = math.asin(min(1.0, disc.radius / distance))
    return bearing, half_width


def rebase_overlapping_history(
    body: FlyBody,
    previous: tuple[VisualDisc, ...],
    current: tuple[VisualDisc, ...],
) -> tuple[VisualDisc, ...]:
    """Reset only invalid temporal baselines after ego-motion crosses a disc.

    Scenes are stored in world coordinates, while the retinal projection is
    evaluated at the current body pose.  A fast body step can therefore make
    an otherwise valid previous disc overlap the body.  Reusing the current
    disc for that channel gives a conservative zero-motion baseline; the
    strict validation in :func:`observe_retina` remains unchanged.
    """

    if len(previous) != len(current):
        raise ValueError("previous and current visual scenes must have equal length")
    rebased: list[VisualDisc] = []
    for old_disc, new_disc in zip(previous, current, strict=True):
        distance = math.hypot(old_disc.x - body.x, old_disc.y - body.y)
        rebased.append(new_disc if distance <= old_disc.radius else old_disc)
    return tuple(rebased)


def observe_retina(
    body: FlyBody,
    previous: tuple[VisualDisc, ...],
    current: tuple[VisualDisc, ...],
) -> RetinalObservation:
    """Project paired anonymous discs into bounded left/right temporal features."""

    if len(previous) != len(current):
        raise ValueError("previous and current visual scenes must have equal length")
    left_luminance = right_luminance = 0.0
    left_contrast = right_contrast = 0.0
    left_motion = right_motion = 0.0
    looming = 0.0
    for old_disc, new_disc in zip(previous, current, strict=True):
        old_bearing, old_width = _disc_features(body, old_disc)
        new_bearing, new_width = _disc_features(body, new_disc)
        luminance = min(1.0, 2.0 * new_width / math.pi)
        contrast = min(1.0, abs(new_width - old_width) * 2.0 / math.pi)
        motion = min(1.0, abs(_angular_delta(new_bearing, old_bearing)) / math.pi)
        expansion = min(1.0, max(0.0, new_width - old_width) * 2.0 / math.pi)
        if new_bearing > 0.0:
            left_luminance = max(left_luminance, luminance)
            left_contrast = max(left_contrast, contrast)
            left_motion = max(left_motion, motion)
        elif new_bearing < 0.0:
            right_luminance = max(right_luminance, luminance)
            right_contrast = max(right_contrast, contrast)
            right_motion = max(right_motion, motion)
        looming = max(looming, expansion)
    return RetinalObservation(
        left_luminance,
        right_luminance,
        left_contrast,
        right_contrast,
        left_motion,
        right_motion,
        looming,
    )


def _validate_ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique IDs")
    return values


@dataclass(frozen=True)
class VisualInterfaceMap:
    """Resolved hemispheric banks for photoreceptors and feature calibrations."""

    left_r1_r6_ids: tuple[int, ...]
    right_r1_r6_ids: tuple[int, ...]
    left_hs_ids: tuple[int, ...]
    right_hs_ids: tuple[int, ...]
    left_lc16_ids: tuple[int, ...]
    right_lc16_ids: tuple[int, ...]

    def validate_disjoint_nonempty(self) -> None:
        groups = tuple(
            _validate_ids(getattr(self, name), name)
            for name in (
                "left_r1_r6_ids",
                "right_r1_r6_ids",
                "left_hs_ids",
                "right_hs_ids",
                "left_lc16_ids",
                "right_lc16_ids",
            )
        )
        flattened = [value for group in groups for value in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("visual interface populations must be disjoint")


@dataclass(frozen=True)
class VisualLoomingMap:
    """Exact projection and descending populations for a looming calibration."""

    left_lc4_ids: tuple[int, ...]
    right_lc4_ids: tuple[int, ...]
    left_lplc2_ids: tuple[int, ...]
    right_lplc2_ids: tuple[int, ...]
    left_dnp01_ids: tuple[int, ...]
    right_dnp01_ids: tuple[int, ...]
    left_dnp02_ids: tuple[int, ...]
    right_dnp02_ids: tuple[int, ...]

    def validate_disjoint_nonempty(self) -> None:
        names = (
            "left_lc4_ids",
            "right_lc4_ids",
            "left_lplc2_ids",
            "right_lplc2_ids",
            "left_dnp01_ids",
            "right_dnp01_ids",
            "left_dnp02_ids",
            "right_dnp02_ids",
        )
        groups = tuple(_validate_ids(getattr(self, name), name) for name in names)
        flattened = [value for group in groups for value in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("looming interface populations must be disjoint")

    def populations(self) -> dict[str, tuple[int, ...]]:
        self.validate_disjoint_nonempty()
        return {
            "lc4_left": self.left_lc4_ids,
            "lc4_right": self.right_lc4_ids,
            "lplc2_left": self.left_lplc2_ids,
            "lplc2_right": self.right_lplc2_ids,
            "dnp01_left": self.left_dnp01_ids,
            "dnp01_right": self.right_dnp01_ids,
            "dnp02_left": self.left_dnp02_ids,
            "dnp02_right": self.right_dnp02_ids,
        }


class VisualInterfaceEncoder:
    """Encode retinal features while preserving the photoreceptor/feature boundary."""

    version = "hemispheric-visual-interface-v1"

    def __init__(self, mapping: VisualInterfaceMap, *, total_voltage: float = 68.75) -> None:
        if not math.isfinite(total_voltage) or total_voltage <= 0:
            raise ValueError("total_voltage must be finite and positive")
        mapping.validate_disjoint_nonempty()
        self.mapping = mapping
        self.total_voltage = float(total_voltage)

    def encode_photoreceptors(
        self, observation: RetinalObservation, *, step: int
    ) -> tuple[ExternalEvent, ...]:
        if step < 0:
            raise ValueError("step must be non-negative")
        channels = (
            ("luminance_left", self.mapping.left_r1_r6_ids, observation.left_luminance),
            ("luminance_right", self.mapping.right_r1_r6_ids, observation.right_luminance),
            ("contrast_left", self.mapping.left_r1_r6_ids, observation.left_contrast),
            ("contrast_right", self.mapping.right_r1_r6_ids, observation.right_contrast),
        )
        return self._events(channels, step)

    def encode_photoreceptor_spikes(
        self,
        observation: RetinalObservation,
        *,
        steps: int,
        seed: int,
        rate_hz: float = 150.0,
        dt_ms: float = 0.1,
    ) -> tuple[ExternalEvent, ...]:
        """Encode luminance/contrast as deterministic source-equivalent spikes.

        ``encode_photoreceptors`` is a population-voltage calibration API.  The Shiu
        simulator consumes voltage jumps per source event, so dividing one population
        voltage across every cell leaves each R1-R6 neuron below threshold.  This API
        preserves the same anonymous hemispheric boundary while generating sparse
        per-neuron source events at a declared rate.
        """

        if type(steps) is not int or steps <= 0:
            raise ValueError("photoreceptor spike steps must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("photoreceptor spike seed must be a non-negative integer")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("photoreceptor spike rate must be finite and positive")
        if not math.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("photoreceptor spike timestep must be finite and positive")
        probability = rate_hz * dt_ms / 1000.0
        if probability > 1.0:
            raise ValueError("photoreceptor spike probability must not exceed one")

        generator = np.random.default_rng(seed)
        banks = (
            (
                "photoreceptor_spikes_left",
                self.mapping.left_r1_r6_ids,
                max(observation.left_luminance, observation.left_contrast),
            ),
            (
                "photoreceptor_spikes_right",
                self.mapping.right_r1_r6_ids,
                max(observation.right_luminance, observation.right_contrast),
            ),
        )
        events: list[ExternalEvent] = []
        for step in range(steps):
            for channel, neuron_ids, drive in banks:
                selected = generator.random(len(neuron_ids)) < probability * drive
                if not np.any(selected):
                    continue
                selected_ids = tuple(
                    neuron_id
                    for neuron_id, include in zip(neuron_ids, selected, strict=True)
                    if include
                )
                events.append(
                    ExternalEvent(
                        step=step,
                        neuron_ids=selected_ids,
                        voltages=(self.total_voltage,) * len(selected_ids),
                        channel=channel,
                    )
                )
        return tuple(events)

    def encode_feature_calibration(
        self, observation: RetinalObservation, *, step: int
    ) -> tuple[ExternalEvent, ...]:
        if step < 0:
            raise ValueError("step must be non-negative")
        channels = (
            ("hs_optic_flow_left", self.mapping.left_hs_ids, observation.left_motion),
            ("hs_optic_flow_right", self.mapping.right_hs_ids, observation.right_motion),
            ("lc16_looming_left", self.mapping.left_lc16_ids, observation.looming),
            ("lc16_looming_right", self.mapping.right_lc16_ids, observation.looming),
        )
        return self._events(channels, step)

    def _events(
        self,
        channels: tuple[tuple[str, tuple[int, ...], float], ...],
        step: int,
    ) -> tuple[ExternalEvent, ...]:
        return tuple(
            ExternalEvent(
                step=step,
                neuron_ids=neuron_ids,
                voltages=tuple(self.total_voltage * value / len(neuron_ids) for _ in neuron_ids),
                channel=channel,
            )
            for channel, neuron_ids, value in channels
            if value > 0.0
        )
