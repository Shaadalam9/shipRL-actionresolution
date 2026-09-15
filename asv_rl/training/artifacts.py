"""Output layout for a training run.

Every run writes the same set of files, so a trained policy always arrives with
the exact config that produced it::

    <run directory>/
    |-- config.yaml            resolved configuration, including overrides
    |-- metrics.json           headline numbers for quick comparison
    |-- training_history.json  per-episode returns and losses
    |-- policy/                final policy, ready for evaluate.py
    |-- checkpoints/ep_<n>/    intermediate policies
    `-- plots/                 return and loss curves, as PNG, HTML and PDF
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def moving_average(values, window: int):
    """Trailing moving average, returned as a list.

    Returns an empty list when there is not yet a full window of data, which
    keeps short smoke-test runs from producing misleading curves.
    """
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.tolist()
    if len(values) < window:
        return []
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


class RunArtifacts:
    """Creates and populates the output directory for one training run."""

    def __init__(self, root: str | Path, config):
        self.root = Path(root)
        self.plots_dir = self.root / "plots"
        self.checkpoints_dir = self.root / "checkpoints"
        self.policy_dir = self.root / "policy"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        config.save(self.root / "config.yaml")

    # -- policies ----------------------------------------------------------

    def save_checkpoint(self, agent, step: int) -> Path:
        """Save an intermediate policy under ``checkpoints/ep_<step>``."""
        from tf_agents.policies import policy_saver

        path = self.checkpoints_dir / f"ep_{step}"
        policy_saver.PolicySaver(agent.policy).save(str(path))
        return path

    def save_final_policy(self, agent) -> Path:
        """Save the policy that evaluate.py loads by default."""
        from tf_agents.policies import policy_saver

        policy_saver.PolicySaver(agent.policy).save(str(self.policy_dir))
        return self.policy_dir

    # -- metrics -----------------------------------------------------------

    def save_history(self, returns, losses, window: int):
        """Write raw and smoothed training curves as JSON."""
        history = {
            "returns": [float(v) for v in returns],
            "loss": [float(v) for v in losses],
            "avg_returns": moving_average(returns, window),
            "avg_loss": moving_average(losses, window),
            "metric_window": window,
        }
        with open(self.root / "training_history.json", "w") as fid:
            json.dump(history, fid, indent=2)
        return history

    def save_metrics(self, returns, losses, window: int, extra: dict | None = None):
        """Write the headline numbers used to compare runs."""
        returns = np.asarray(returns, dtype=np.float64)
        tail = returns[-window:] if len(returns) >= window else returns
        metrics = {
            "algorithm": self.config.algorithm,
            "run_name": self.config.run_name,
            "seed": self.config.seed,
            "episodes": int(len(returns)),
            "final_return_mean": float(np.mean(tail)) if len(tail) else None,
            "final_return_std": float(np.std(tail)) if len(tail) else None,
            "best_return": float(np.max(returns)) if len(returns) else None,
            "final_loss": float(losses[-1]) if len(losses) else None,
            "metric_window": window,
        }
        metrics.update(extra or {})
        with open(self.root / "metrics.json", "w") as fid:
            json.dump(metrics, fid, indent=2)
        return metrics

    def save_curves(self, returns, losses, window: int):
        """Write smoothed return and loss curves as PNG, HTML and PDF."""
        from asv_rl.evaluation.plotting import save_figure, training_curve_figure

        written = []
        for values, y_title, stem in (
            (moving_average(returns, window), f"Return ({window}-episode moving average)", "returns"),
            (moving_average(losses, window), f"Loss ({window}-episode moving average)", "loss"),
        ):
            if not len(values):
                continue
            fig = training_curve_figure(values, y_title)
            written.extend(save_figure(fig, self.plots_dir / stem))
        return written
