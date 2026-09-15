#!/usr/bin/env python3
"""Train an agent from a YAML experiment config.

Examples::

    # Train each algorithm with its published settings
    uv run scripts/train.py configs/ddpg.yaml
    uv run scripts/train.py configs/dqn_5_action.yaml

    # A short smoke test before committing to a full run
    uv run scripts/train.py configs/ppo.yaml --set training.iterations=2 \
        --set training.episodes_per_iteration=2 --run-name smoke_test

    # A seed sweep, replacing the old batch_train.py
    for seed in 12345 43738 95832; do
        uv run scripts/train.py configs/dqn_5_action.yaml \
            --set seed=$seed --run-name "dqn5_seed_$seed"
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_rl.config import ConfigError  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to a YAML experiment config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config option, e.g. --set training.episodes=500. Repeatable.",
    )
    parser.add_argument(
        "--run-name",
        help="Name of the output run directory (default: run_name from the config)",
    )
    parser.add_argument(
        "--experiment",
        help="Name of the model subdirectory grouping this run (default: experiment from the config)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory that holds all runs (default: ./models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved config without training",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from asv_rl.agents.registry import load_config

    try:
        config, spec = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.run_name:
        config.run_name = args.run_name
    if args.experiment:
        config.experiment = args.experiment

    output_dir = Path(args.output_root) / config.experiment / config.run_name

    if args.dry_run:
        import yaml

        print(yaml.safe_dump(config.as_dict(), sort_keys=False, default_flow_style=False))
        print(f"# would train {spec.name} ({spec.description})")
        print(f"# output directory: {output_dir}")
        return 0

    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Refusing to overwrite the existing run at {output_dir}.\n"
            "Pass --run-name to choose a different name, or delete that directory.",
            file=sys.stderr,
        )
        return 1

    import tensorflow as tf

    from asv_rl.agents.registry import OFF_POLICY
    from asv_rl.training.loops import train_off_policy, train_on_policy

    # Seeding both TensorFlow and NumPy; op determinism makes repeated runs with
    # the same seed reproducible at some cost in speed.
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()

    print(f"Algorithm : {spec.name} ({spec.description})")
    print(f"Run name  : {config.run_name}")
    print(f"Seed      : {config.seed}")
    print(f"Output    : {output_dir}")
    print("-" * 72, flush=True)

    trainer = train_off_policy if spec.trainer == OFF_POLICY else train_on_policy
    metrics = trainer(config, spec, output_dir)

    print("-" * 72)
    print("Training complete.")
    print(f"  final return (mean of last {metrics['metric_window']}): {metrics['final_return_mean']}")
    print(f"  policy   : {output_dir / 'policy'}")
    print(f"  metrics  : {output_dir / 'metrics.json'}")
    print(f"  evaluate : uv run scripts/evaluate.py {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
