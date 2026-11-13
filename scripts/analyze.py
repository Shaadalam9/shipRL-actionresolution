#!/usr/bin/env python3
"""Run every trained controller on the same reference paths and compare them.

This is the headline experiment of the repository: the controllers listed
under ``agents`` in the root ``config`` file are driven over identical paths
under identical conditions, and the resulting tracks, errors, and summary
metrics are plotted together.

Every figure is written as PNG, HTML, and PDF. The HTML version is interactive
-- pan, zoom, and hover to read values off any track.

Examples::

    # Every controller named in the config file, on the default reference paths
    uv run scripts/analyze.py

    # One path, in wind
    uv run scripts/analyze.py --trajectory figure_eight --wind-speed 6 --wind-dir 90

    # Compare only two controllers, with a specific run of each
    uv run scripts/analyze.py \
        --model ddpg=models/ddpg/calm_water_v1 \
        --model ppo=models/ppo/calm_water_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_rl import settings  # noqa: E402
from asv_rl.config import ConfigError  # noqa: E402
from asv_rl.evaluation.trajectories import (  # noqa: E402
    EVALUATION_ARRIVAL_RADIUS,
    PAPER_CALM_PATHS,
    parse_arrival_radius,
)

#: Reference paths compared when --trajectory is not given.
DEFAULT_TRAJECTORIES = PAPER_CALM_PATHS

#: Preferred run when an experiment directory holds more than one.
PREFERRED_RUN = "calm_water_v1"


def discover_models(models_root: Path) -> dict[str, Path]:
    """Find one usable run directory per experiment under ``models/``.

    A run is usable when it has both a ``config.yaml`` and a ``policy/``.
    Only the controllers listed under ``agents`` in the root ``config`` file
    are kept, and they are returned in that order, so the comparison is
    changed by editing the config rather than the code. An empty or missing
    ``agents`` list means every run found.
    """
    found: dict[str, list[Path]] = {}
    for config_path in sorted(models_root.glob("*/*/config.yaml")):
        run_dir = config_path.parent
        if not (run_dir / "policy").exists():
            continue
        found.setdefault(run_dir.parent.name, []).append(run_dir)

    selected = {}
    for experiment, runs in found.items():
        preferred = [r for r in runs if r.name == PREFERRED_RUN]
        selected[experiment] = preferred[0] if preferred else runs[0]
        if len(runs) > 1:
            print(
                f"note: {experiment} has {len(runs)} runs; using {selected[experiment].name}. "
                f"Pass --model {experiment}=<path> to choose another.",
                file=sys.stderr,
            )

    wanted = settings.get("agents") or []
    if wanted:
        missing = [name for name in wanted if name not in selected]
        if missing:
            print(
                f"note: no run found under {models_root} for {', '.join(missing)} "
                "(listed under 'agents' in the config file).",
                file=sys.stderr,
            )
        dropped = [name for name in selected if name not in wanted]
        if dropped:
            print(
                f"note: skipping {', '.join(dropped)}; the config file lists "
                f"{', '.join(wanted)} under 'agents'.",
                file=sys.stderr,
            )
        selected = {name: selected[name] for name in wanted if name in selected}
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        metavar="NAME=PATH",
        help="Controller to include, e.g. --model ddpg=models/ddpg/calm_water_v1. "
             "Repeatable. Overrides the 'agents' list in the config file. "
             "Default: the runs under models/ named by that list.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        dest="trajectories",
        help=f"Reference path to compare on. Repeatable. Default: {', '.join(DEFAULT_TRAJECTORIES)}",
    )
    parser.add_argument("--list-trajectories", action="store_true", help="List reference paths and exit")
    parser.add_argument(
        "--models-root", type=Path, default=REPO_ROOT / "models",
        help="Directory searched for runs (default: ./models)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "analysis",
        help="Where figures and results are written (default: ./analysis)",
    )
    parser.add_argument(
        "--format", action="append", dest="formats",
        choices=["png", "html", "pdf"],
        help="Figure format to write. Repeatable. Default: all three.",
    )
    parser.add_argument(
        "--arrival-radius", type=parse_arrival_radius, default=EVALUATION_ARRIVAL_RADIUS,
        metavar="RADIUS",
        help="Waypoint acceptance radius in ship lengths, applied to every controller so "
             f"the task is identical for all. Default: {EVALUATION_ARRIVAL_RADIUS}, as in the "
             "paper. Pass 'trained' to use each model's own trained value instead.",
    )
    parser.add_argument("--max-steps", type=int, default=5000, help="Step budget per rollout")
    parser.add_argument(
        "--embed-plotlyjs", action="store_true",
        help="Embed plotly.js in the HTML figures (a few MB each) so they render through "
             "htmlpreview, as needed for figures linked from the README.",
    )

    conditions = parser.add_argument_group("environmental conditions (applied to every controller)")
    conditions.add_argument("--wind-speed", type=float, help="Non-dimensional wind speed; enables wind")
    conditions.add_argument("--wind-dir", type=float, default=0.0, help="Wind direction in degrees")
    conditions.add_argument("--wave-height", type=float, help="Wave height; enables waves")
    conditions.add_argument("--wave-period", type=float, default=10.0, help="Wave period in seconds")
    conditions.add_argument("--wave-dir", type=float, default=0.0, help="Wave direction in degrees")
    return parser.parse_args(argv)


def resolve_models(args) -> dict[str, Path]:
    """Build the {name: run directory} mapping from flags or discovery."""
    if args.models:
        models = {}
        for entry in args.models:
            if "=" not in entry:
                raise ConfigError(f"--model {entry!r}: expected the form NAME=PATH")
            name, _, path = entry.partition("=")
            models[name.strip()] = Path(path.strip())
        return models
    return discover_models(args.models_root)


def apply_conditions(env_config, args):
    """Apply the same wind, wave, and arrival radius to every controller."""
    if args.wind_speed is not None:
        env_config.conditions.wind.enabled = True
        env_config.conditions.wind.speed = args.wind_speed
        env_config.conditions.wind.direction_deg = args.wind_dir
    if args.wave_height is not None:
        env_config.conditions.wave.enabled = True
        env_config.conditions.wave.height = args.wave_height
        env_config.conditions.wave.period = args.wave_period
        env_config.conditions.wave.direction_deg = args.wave_dir
    if args.arrival_radius is not None:
        env_config.arrival_radius = args.arrival_radius
    return env_config


def main(argv=None) -> int:
    args = parse_args(argv)
    settings.configure_logging("analyze")

    from asv_rl.evaluation import trajectories as traj_module  # pure geometry, no TF

    if args.list_trajectories:
        print("Available reference paths:")
        for name, factory in sorted(traj_module.TRAJECTORIES.items()):
            summary = (factory.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<18} {summary}")
        return 0

    try:
        models = resolve_models(args)
    except ConfigError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    if not models:
        print(
            f"No runs found under {args.models_root} for the controllers in the "
            "config file. A run directory needs a config.yaml and a policy/ "
            "subdirectory, and its name must be listed under 'agents' in the config.",
            file=sys.stderr,
        )
        return 2

    from asv_rl.agents.registry import load_config
    from asv_rl.evaluation.plotting import (
        FIGURE_FORMATS,
        VALUE_FORMAT,
        grouped_bar_figure,
        label_for,
        save_figure,
        series_figure,
        table_figure,
        track_figure,
    )
    from asv_rl.evaluation.rollout import condition_label, run_rollout

    if args.embed_plotlyjs:
        from functools import partial

        save_figure = partial(save_figure, embed_plotlyjs=True)

    formats = tuple(args.formats) if args.formats else FIGURE_FORMATS

    # Load each controller's environment, then force the same conditions on all.
    configs = {}
    for name, run_dir in models.items():
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            print(f"No config.yaml in {run_dir}", file=sys.stderr)
            return 2
        if not (run_dir / "policy").exists():
            print(f"No saved policy in {run_dir}", file=sys.stderr)
            return 2
        try:
            config, _ = load_config(config_path)
        except ConfigError as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            return 2
        configs[name] = (run_dir, apply_conditions(config.environment, args))

    label = condition_label(next(iter(configs.values()))[1])
    names = args.trajectories or DEFAULT_TRAJECTORIES
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = {n: c.arrival_radius for n, (_, c) in configs.items()}
    print(f"Controllers : {', '.join(label_for(n) for n in configs)}")
    print(f"Paths       : {', '.join(names)}")
    print(f"Conditions  : {label}")
    print(f"Formats     : {', '.join(formats)}")
    print(f"Output      : {output_dir}")
    if len(set(radii.values())) > 1:
        print(
            f"note        : arrival radius differs between controllers ({radii}). "
            "Pass a number to --arrival-radius to make the task identical for all.",
        )
    print("-" * 78, flush=True)

    results = []
    for traj_name in names:
        try:
            path = traj_module.get(traj_name)
        except ValueError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2

        print(f"{traj_name}")
        tracks, errors, rudders = {}, {}, {}

        for name, (run_dir, env_config) in configs.items():
            result, env = run_rollout(
                run_dir / "policy", env_config, path, max_steps=args.max_steps
            )
            tracks[name] = (env.x_traj, env.y_traj)
            steps = list(range(len(env.cross_trk_err_traj)))
            errors[name] = (steps, list(env.cross_trk_err_traj))
            rudders[name] = (
                list(range(len(env.delta_traj))),
                [float(d) * 180 / 3.141592653589793 for d in env.delta_traj],
            )

            row = {
                "trajectory": traj_name,
                "controller": name,
                "run": str(run_dir),
                "cross_track_rmse": result.cross_track_rmse,
                "mean_absolute_rudder_deg": result.mean_absolute_rudder_deg,
                "waypoints_reached": result.waypoints_reached,
                "waypoints_total": result.waypoints_total,
                "steps": result.steps,
                "completed": result.completed,
            }
            results.append(row)
            status = "completed" if result.completed else (
                f"stopped at {result.waypoints_reached}/{result.waypoints_total - 1}"
            )
            print(
                f"  {label_for(name):<20} xte_rmse={result.cross_track_rmse:8.4f}  "
                f"mean|rudder|={result.mean_absolute_rudder_deg:5.2f}deg  "
                f"steps={result.steps:<5d} {status}",
                flush=True,
            )

        save_figure(
            track_figure(
                path.x_reference, path.y_reference, path.x_waypoints, path.y_waypoints,
                tracks,
            ),
            output_dir / f"track_{traj_name}", formats,
        )
        save_figure(
            series_figure(
                errors, x_title="Step",
                y_title="Cross-track error (ship lengths)", zero_line=True,
            ),
            output_dir / f"cross_track_error_{traj_name}", formats,
        )
        save_figure(
            series_figure(
                rudders, x_title="Step",
                y_title="Realised rudder angle (degrees)", zero_line=True,
            ),
            output_dir / f"rudder_angle_{traj_name}", formats,
        )
        print(f"  figures -> {output_dir}/*_{traj_name}.{{{','.join(formats)}}}", flush=True)

    # Summary across every path
    controllers = list(configs)
    pretty_names = [n.replace("_", " ").title() for n in names]

    def column(metric):
        return {
            c: [next((r[metric] for r in results if r["trajectory"] == t and r["controller"] == c), None)
                for t in names]
            for c in controllers
        }

    save_figure(
        grouped_bar_figure(
            pretty_names, column("cross_track_rmse"),
            y_title="Cross-track RMSE (ship lengths)",
        ),
        output_dir / "summary_cross_track_rmse", formats,
    )
    save_figure(
        grouped_bar_figure(
            pretty_names, column("mean_absolute_rudder_deg"),
            y_title="Mean absolute rudder angle (degrees)",
        ),
        output_dir / "summary_rudder_effort", formats,
    )
    save_figure(
        table_figure(
            ["Path", "Controller", "Cross-track RMSE", "Mean |rudder| (deg)", "Steps", "Completed"],
            [
                [r["trajectory"] for r in results],
                [label_for(r["controller"]) for r in results],
                [f"{r['cross_track_rmse']:{VALUE_FORMAT}}" for r in results],
                [f"{r['mean_absolute_rudder_deg']:{VALUE_FORMAT}}" for r in results],
                [r["steps"] for r in results],
                ["yes" if r["completed"] else "no" for r in results],
            ],
        ),
        output_dir / "summary_table", formats,
    )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as fid:
        json.dump(
            {"conditions": label, "arrival_radius": radii, "results": results},
            fid, indent=2,
        )

    print("-" * 78)
    print(f"Compared {len(controllers)} controllers over {len(names)} paths.")
    print(f"  figures : {output_dir}")
    print(f"  results : {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
