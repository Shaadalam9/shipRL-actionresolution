"""Reference paths used to evaluate a trained controller.

Each generator returns a :class:`ReferencePath`: the waypoints the agent must
hit, a densely sampled version of the same curve for plotting, and the initial
heading the ship starts from.

Coordinates are non-dimensional, in ship lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferencePath:
    """A path to follow, plus the heading the ship starts with."""

    name: str
    x_waypoints: np.ndarray
    y_waypoints: np.ndarray
    initial_heading: float
    x_reference: np.ndarray
    y_reference: np.ndarray

    @property
    def num_waypoints(self) -> int:
        return len(self.x_waypoints)


def _dense(n: int = 1000):
    return np.linspace(0, 2 * np.pi, num=n)


def ellipse(direction: str = "starboard", semi_major: float = 14.0, semi_minor: float = 12.0, npoints: int = 15):
    """An ellipse, entered tangentially."""
    sign = 1 if direction == "starboard" else -1
    theta = np.linspace(0, sign * 2 * np.pi, num=npoints, endpoint=True)
    theta_ref = np.linspace(0, sign * 2 * np.pi, num=1000)
    return ReferencePath(
        name=f"ellipse_{direction}",
        x_waypoints=semi_major * np.cos(theta),
        y_waypoints=semi_minor * np.sin(theta),
        initial_heading=sign * np.pi / 2,
        x_reference=semi_major * np.cos(theta_ref),
        y_reference=semi_minor * np.sin(theta_ref),
    )


def cardioid(direction: str = "starboard", scale: float = 10.0, npoints: int = 18):
    """A cardioid, which forces a tight cusp turn."""
    sign = 1 if direction == "starboard" else -1
    theta = np.linspace(0, sign * 2 * np.pi, num=npoints, endpoint=True)
    theta_ref = np.linspace(0, sign * 2 * np.pi, num=1000)
    return ReferencePath(
        name=f"cardioid_{direction}",
        x_waypoints=scale * (2 * np.cos(theta) - np.cos(2 * theta)),
        y_waypoints=scale * (2 * np.sin(theta) - np.sin(2 * theta)),
        initial_heading=0.0,
        x_reference=scale * (2 * np.cos(theta_ref) - np.cos(2 * theta_ref)),
        y_reference=scale * (2 * np.sin(theta_ref) - np.sin(2 * theta_ref)),
    )


def astroid(direction: str = "starboard", scale: float = 20.0, npoints: int = 18):
    """An astroid: four cusps, the hardest of the closed reference paths."""
    sign = 1 if direction == "starboard" else -1
    theta = np.linspace(0, sign * 2 * np.pi, num=npoints, endpoint=True)
    theta_ref = np.linspace(0, sign * 2 * np.pi, num=1000)
    return ReferencePath(
        name=f"astroid_{direction}",
        x_waypoints=scale * np.cos(theta) ** 3,
        y_waypoints=scale * np.sin(theta) ** 3,
        initial_heading=0.0,
        x_reference=scale * np.cos(theta_ref) ** 3,
        y_reference=scale * np.sin(theta_ref) ** 3,
    )


def spiral(radius: float = 15.0, turns: int = 4, npoints: int = 40):
    """An outward spiral, traversed from the outside in."""
    rad = np.linspace(0, radius, num=npoints, endpoint=True)
    rad_ref = np.linspace(0, radius, num=1000)
    theta = np.linspace(0, 2 * np.pi * turns, num=npoints, endpoint=True)
    theta_ref = np.linspace(0, 2 * np.pi * turns, num=1000)
    return ReferencePath(
        name="spiral",
        x_waypoints=(rad * np.cos(theta))[::-1],
        y_waypoints=(rad * np.sin(theta))[::-1],
        initial_heading=-np.pi / 2,
        x_reference=rad_ref * np.cos(theta_ref),
        y_reference=rad_ref * np.sin(theta_ref),
    )


def straight(length: float = 128.0, npoints: int = 9):
    """A straight line: the baseline course-keeping test."""
    x_wp = np.linspace(0, length, num=npoints, endpoint=True)
    y_wp = np.zeros_like(x_wp)
    return ReferencePath(
        name="straight",
        x_waypoints=x_wp,
        y_waypoints=y_wp,
        initial_heading=0.0,
        x_reference=x_wp,
        y_reference=y_wp,
    )


def sine_wave(amplitude: float = 20.0, frequency: float = 5.0, stretch: float = 15.0, npoints: int = 40):
    """A sinusoid, which exercises repeated course reversals."""
    theta = np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)
    theta_ref = _dense()
    return ReferencePath(
        name="sine_wave",
        x_waypoints=stretch * theta,
        y_waypoints=amplitude * np.sin(frequency * theta),
        initial_heading=np.pi / 2,
        x_reference=stretch * theta_ref,
        y_reference=amplitude * np.sin(frequency * theta_ref),
    )


def star():
    """A five-pointed star traced by its chords: sharp alternating turns."""
    x_wp = np.array([0, 5.878, -9.511, 9.511, -5.878, 0])
    y_wp = np.array([10, -8.09, 3.09, 3.09, -8.09, 10])
    return ReferencePath(
        name="star",
        x_waypoints=x_wp,
        y_waypoints=y_wp,
        initial_heading=float(np.arctan2(y_wp[1] - y_wp[0], x_wp[1] - x_wp[0])),
        x_reference=x_wp,
        y_reference=y_wp,
    )


def figure_eight(start: str = "bottom", circle_radius: float = 9.0, npoints: int = 15):
    """A figure of eight: two circles of opposite turning direction."""
    upper = npoints - npoints // 2
    lower = npoints // 2

    theta1 = np.linspace(-0.5 * np.pi, 1.5 * np.pi, upper, endpoint=True)
    theta1_ref = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000)
    x1 = circle_radius * np.cos(theta1)
    y1 = circle_radius * np.sin(theta1) + circle_radius
    x1_ref = circle_radius * np.cos(theta1_ref)
    y1_ref = circle_radius * np.sin(theta1_ref) + circle_radius

    if start == "bottom":
        theta2 = np.linspace(0.5 * np.pi, 2.5 * np.pi, lower, endpoint=False)
        theta2_ref = np.linspace(0.5 * np.pi, 2.5 * np.pi, 1000)
        x2 = circle_radius * np.cos(theta2)
        y2 = circle_radius * np.sin(theta2) - circle_radius
        x2_ref = circle_radius * np.cos(theta2_ref)
        y2_ref = circle_radius * np.sin(theta2_ref) - circle_radius

        x_wp, y_wp = np.append(x1, x2[::-1]), np.append(y1, y2[::-1])
        x_ref, y_ref = np.append(x1_ref, x2_ref[::-1]), np.append(y1_ref, y2_ref[::-1])
    else:
        theta2 = np.linspace(2.5 * np.pi, 0.5 * np.pi, lower, endpoint=False)
        theta2_ref = np.linspace(2.5 * np.pi, 0.5 * np.pi, 1000)
        x2 = circle_radius * np.cos(theta2)
        y2 = circle_radius * np.sin(theta2) - circle_radius
        x2_ref = circle_radius * np.cos(theta2_ref)
        y2_ref = circle_radius * np.sin(theta2_ref) - circle_radius

        x_wp, y_wp = np.append(x2, x1), np.append(y2, y1)
        x_ref, y_ref = np.append(x2_ref, x1_ref[::-1]), np.append(y2_ref, y1_ref[::-1])

    return ReferencePath(
        name=f"figure_eight_{start}",
        x_waypoints=x_wp,
        y_waypoints=y_wp,
        initial_heading=0.0,
        x_reference=x_ref,
        y_reference=y_ref,
    )


def single_waypoint(quadrant: int = 1, distance: float = 10.0):
    """A single goal in one quadrant: the simplest possible check."""
    if quadrant not in (1, 2, 3, 4):
        raise ValueError(f"quadrant must be 1-4, got {quadrant}")
    signs = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}[quadrant]
    x_wp = np.array([0.0, signs[0] * distance])
    y_wp = np.array([0.0, signs[1] * distance])
    return ReferencePath(
        name=f"single_waypoint_q{quadrant}",
        x_waypoints=x_wp,
        y_waypoints=y_wp,
        initial_heading=0.0,
        x_reference=x_wp,
        y_reference=y_wp,
    )


#: Reference paths selectable by name from ``evaluate.py``.
TRAJECTORIES = {
    "ellipse": ellipse,
    "cardioid": cardioid,
    "astroid": astroid,
    "spiral": spiral,
    "straight": straight,
    "sine_wave": sine_wave,
    "star": star,
    "figure_eight": figure_eight,
    "single_waypoint": single_waypoint,
}


def get(name: str, **kwargs) -> ReferencePath:
    """Build a reference path by name, e.g. ``get("ellipse", direction="port")``."""
    try:
        factory = TRAJECTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown trajectory {name!r}; available: {sorted(TRAJECTORIES)}"
        ) from None
    return factory(**kwargs)
