"""Reference paths and rollout helpers for evaluating a trained policy.

``trajectories`` is pure geometry and imports nothing heavy, so the rollout
helpers -- which pull in TensorFlow -- are re-exported lazily. Listing or
building a reference path therefore works without a TensorFlow install.
"""

from asv_rl.evaluation.trajectories import TRAJECTORIES, ReferencePath, get

__all__ = ["TRAJECTORIES", "ReferencePath", "get", "RolloutResult", "run_rollout", "plot_rollout"]

_LAZY = {"RolloutResult", "run_rollout", "plot_rollout"}


def __getattr__(name):
    if name in _LAZY:
        from asv_rl.evaluation import rollout

        return getattr(rollout, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
