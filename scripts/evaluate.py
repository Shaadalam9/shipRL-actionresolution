#!/usr/bin/env python3
"""Evaluate a trained policy on one or more reference paths.

A run directory produced by ``train.py`` carries its own ``config.yaml``, so
evaluation reuses the exact environment the policy was trained against unless
you override the conditions on the command line.

Examples::

    # Every reference path, in calm water
    uv run scripts/evaluate.py models/ddpg/calm_water_v1

    # One path, in wind
    uv run scripts/evaluate.py models/ppo/calm_water_v1 \
        --trajectory figure_eight --wind-speed 6 --wind-dir 90

    # In combined wind and waves
    uv run scripts/evaluate.py models/dqn_5_action/calm_water_v1 \
        --trajectory ellipse --wind-speed 6 --wind-dir 90 \
        --wave-height 3 --wave-period 10 --wave-dir 45

Figures are written as PNG, HTML, and PDF. To compare all four controllers on
the same paths in one go, use scripts/analyze.py instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_rl.config import ConfigError  # noqa: E402

#: Paths evaluated when --trajectory is not given.
DEFAULT_TRAJECTORIES = ["straight", "ellipse", "cardioid", "figure_eight", "star"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Optional so that --list-trajectories works on its own; main() checks it.
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="A run directory containing config.yaml and policy/",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        help="Policy to load (default: <run_dir>/policy). Use to evaluate an intermediate checkpoint.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        dest="trajectories",
        help=f"Reference path to follow. Repeatable. Default: {', '.join(DEFAULT_TRAJECTORIES)}",
    )
    parser.add_argument("--list-trajectories", action="store_true", help="List reference paths and exit")
    parser.add_argument("--output-dir", type=Path, help="Where to write plots (default: <run_dir>/evaluation)")
    parser.add_argument("--max-steps", type=int, default=5000, help="Step budget per rollout")
    parser.add_argument(
        "--format", action="append", dest="formats", choices=["png", "html", "pdf"],
        help="Figure format to write. Repeatable. Default: all three.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-waypoint progress")

    conditions = parser.add_argument_group("environmental conditions (override the training config)")
    conditions.add_argument("--wind-speed", type=float, help="Non-dimensional wind speed; enables wind")
    conditions.add_argument("--wind-dir", type=float, default=0.0, help="Wind direction in degrees")
    conditions.add_argument("--wave-height", type=float, help="Wave height; enables waves")
    conditions.add_argument("--wave-period", type=float, default=10.0, help="Wave period in seconds")
    conditions.add_argument("--wave-dir", type=float, default=0.0, help="Wave direction in degrees")
    return parser.parse_args(argv)


def apply_conditions(env_config, args):
    """Let command-line flags override the wind and wave state of the config."""
    if args.wind_speed is not None:
        env_config.conditions.wind.enabled = True
        env_config.conditions.wind.speed = args.wind_speed
        env_config.conditions.wind.direction_deg = args.wind_dir
    if args.wave_height is not None:
        env_config.conditions.wave.enabled = True
        env_config.conditions.wave.height = args.wave_height
        env_config.conditions.wave.period = args.wave_period
        env_config.conditions.wave.direction_deg = args.wave_dir
    return env_config


def main(argv=None) -> int:
    args = parse_args(argv)

    from asv_rl.evaluation import trajectories as traj_module  # pure geometry, no TF

    if args.list_trajectories:
        print("Available reference paths:")
        for name, factory in sorted(traj_module.TRAJECTORIES.items()):
            summary = (factory.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<18} {summary}")
        return 0

    if args.run_dir is None:
        print(
            "A run directory is required, e.g. models/ddpg/calm_water_v1\n"
            "Use --list-trajectories to see the available reference paths.",
            file=sys.stderr,
        )
        return 2

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(
            f"No config.yaml in {run_dir}. Point this at a run directory created by train.py.",
            file=sys.stderr,
        )
        return 2

    policy_dir = args.policy or (run_dir / "policy")
    if not policy_dir.exists():
        print(f"No saved policy at {policy_dir}", file=sys.stderr)
        return 2

    from asv_rl.agents.registry import load_config

    try:
        config, _ = load_config(config_path)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    env_config = apply_conditions(config.environment, args)

    from asv_rl.evaluation.plotting import FIGURE_FORMATS
    from asv_rl.evaluation.rollout import (
        condition_label,
        plot_rollout,
        plot_rollout_errors,
        run_rollout,
        save_results,
    )

    formats = tuple(args.formats) if args.formats else FIGURE_FORMATS
    output_dir = args.output_dir or (run_dir / "evaluation")
    names = args.trajectories or DEFAULT_TRAJECTORIES
    label = condition_label(env_config)

    print(f"Policy     : {policy_dir}")
    print(f"Conditions : {label}")
    print(f"Formats    : {', '.join(formats)}")
    print("-" * 72, flush=True)

    results = []
    for name in names:
        try:
            path = traj_module.get(name)
        except ValueError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2

        result, env = run_rollout(
            policy_dir, env_config, path, max_steps=args.max_steps, verbose=args.verbose
        )
        results.append(result)

        written = plot_rollout(
            env, path, output_dir / f"track_{path.name}",
            algorithm=config.experiment, formats=formats,
        )
        written += plot_rollout_errors(
            env, output_dir / f"cross_track_error_{path.name}",
            algorithm=config.experiment, formats=formats,
        )
        status = "completed" if result.completed else (
            f"stopped at waypoint {result.waypoints_reached}/{result.waypoints_total - 1}"
        )
        print(
            f"{name:<18} xte_rmse={result.cross_track_rmse:7.4f}  "
            f"mean|rudder|={result.mean_absolute_rudder_deg:5.2f}deg  "
            f"steps={result.steps:<5d} {status}",
            flush=True,
        )
        print(f"{'':<18} figures -> {len(written)} file(s) in {output_dir}")

    results_path = save_results(results, output_dir / "results.json")
    print("-" * 72)
    print(f"Wrote {len(results)} result(s) to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
