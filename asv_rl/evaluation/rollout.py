"""Run a saved policy against a reference path and report tracking error."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tf_agents.environments import tf_py_environment, wrappers

from asv_rl.config import EnvironmentConfig
from asv_rl.envs.ship_env import EVALUATE, ShipEnv
from asv_rl.evaluation.plotting import (
    FIGURE_FORMATS,
    color_for,
    save_figure,
    series_figure,
    track_figure,
)
from asv_rl.evaluation.trajectories import ReferencePath

#: Generous step budget: an evaluation episode ends when the last waypoint is
#: reached, not when a training-length time limit expires.
DEFAULT_MAX_STEPS = 5000


@dataclass
class RolloutResult:
    """Tracking performance of one policy on one reference path."""

    trajectory: str
    policy: str
    cross_track_rmse: float
    mean_absolute_rudder_deg: float
    waypoints_reached: int
    waypoints_total: int
    steps: int

    @property
    def completed(self) -> bool:
        return self.waypoints_reached == self.waypoints_total - 1


def run_rollout(
    policy_dir: str | Path,
    env_config: EnvironmentConfig,
    path: ReferencePath,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = False,
) -> tuple[RolloutResult, ShipEnv]:
    """Drive ``path`` with the saved policy at ``policy_dir``.

    Returns the summary metrics and the environment itself, which still holds
    the full recorded trajectory for plotting.
    """
    policy_dir = Path(policy_dir)
    if not policy_dir.exists():
        raise FileNotFoundError(f"No saved policy at {policy_dir}")

    initial_state = [
        env_config.initial_surge, 0, 0,
        float(path.x_waypoints[0]), float(path.y_waypoints[0]),
        float(path.initial_heading), 0,
    ]

    env = ShipEnv(
        env_config,
        mode=EVALUATE,
        x_waypoints=path.x_waypoints,
        y_waypoints=path.y_waypoints,
        initial_state=initial_state,
        verbose=verbose,
    )
    tf_env = tf_py_environment.TFPyEnvironment(wrappers.TimeLimit(env, duration=max_steps))

    policy = tf.compat.v2.saved_model.load(str(policy_dir))

    time_step = tf_env.reset()
    steps = 0
    while not np.equal(time_step.step_type, 2):
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)
        steps += 1

    result = RolloutResult(
        trajectory=path.name,
        policy=str(policy_dir),
        cross_track_rmse=env.cross_track_rmse(),
        mean_absolute_rudder_deg=float(np.degrees(env.mean_absolute_rudder())),
        waypoints_reached=env.wp_counter,
        waypoints_total=path.num_waypoints,
        steps=steps,
    )
    return result, env


def hull_outline(x, y, psi, markers: int = 9):
    """Ship outlines drawn along the path, so heading is visible as well as position.

    Returns a single trace's worth of coordinates, with None between outlines to
    break the line rather than joining one hull to the next.
    """
    # Hull shape in ship-fixed coordinates, in ship lengths.
    x_hull = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
    y_hull = 16.1 / 230 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])

    xs, ys = [], []
    if len(x) <= markers:
        return xs, ys
    for i in range(1, markers):
        index = i * (len(x) // markers)
        heading = psi[index]
        xs.extend((x[index] + x_hull * np.cos(heading) - y_hull * np.sin(heading)).tolist() + [None])
        ys.extend((y[index] + x_hull * np.sin(heading) + y_hull * np.cos(heading)).tolist() + [None])
    return xs, ys


def plot_rollout(
    env: ShipEnv,
    path: ReferencePath,
    stem: str | Path,
    algorithm: str = "ddpg",
    formats=FIGURE_FORMATS,
) -> list[Path]:
    """Plot one policy's path against the reference, in PNG, HTML and PDF."""
    x_traj = np.asarray(env.x_traj)
    y_traj = np.asarray(env.y_traj)
    psi_traj = np.asarray(env.psi_traj)

    fig = track_figure(
        path.x_reference, path.y_reference,
        path.x_waypoints, path.y_waypoints,
        {algorithm: (x_traj, y_traj)},
    )

    hull_x, hull_y = hull_outline(x_traj, y_traj, psi_traj)
    if hull_x:
        fig.add_trace(go.Scatter(
            x=hull_x, y=hull_y, mode="lines", name="Ship outline",
            line=dict(color=color_for(algorithm), width=1),
            opacity=0.7, hoverinfo="skip", showlegend=True,
        ))

    return save_figure(fig, stem, formats)


def plot_rollout_errors(
    env: ShipEnv,
    stem: str | Path,
    algorithm: str = "ddpg",
    formats=FIGURE_FORMATS,
) -> list[Path]:
    """Plot cross-track error against time."""
    steps = list(range(len(env.cross_trk_err_traj)))
    fig = series_figure(
        {algorithm: (steps, env.cross_trk_err_traj)},
        x_title="Step", y_title="Cross-track error (ship lengths)",
        zero_line=True,
    )
    return save_figure(fig, stem, formats)


def condition_label(env_config: EnvironmentConfig) -> str:
    """Human-readable description of the wind and wave state, for plot titles."""
    wind, wave = env_config.conditions.wind, env_config.conditions.wave
    parts = []
    if wind.enabled:
        parts.append(f"wind V'={wind.speed:g}, dir={wind.direction_deg:g}°")
    if wave.enabled:
        parts.append(f"waves H={wave.height:g}, T={wave.period:g}s, dir={wave.direction_deg:g}°")
    return "; ".join(parts) if parts else "Calm water"


def save_results(results: list[RolloutResult], output_path: str | Path):
    """Write rollout metrics as JSON for downstream comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fid:
        json.dump([asdict(r) | {"completed": r.completed} for r in results], fid, indent=2)
    return output_path
