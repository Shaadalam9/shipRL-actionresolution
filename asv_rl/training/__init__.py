"""Training loops and run artefacts.

Re-exported lazily so that importing :mod:`asv_rl` does not pull in
TensorFlow until a training loop is actually requested.
"""

__all__ = ["RunArtifacts", "train_off_policy", "train_on_policy"]


def __getattr__(name):
    if name == "RunArtifacts":
        from asv_rl.training.artifacts import RunArtifacts

        return RunArtifacts
    if name in ("train_off_policy", "train_on_policy"):
        from asv_rl.training import loops

        return getattr(loops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
