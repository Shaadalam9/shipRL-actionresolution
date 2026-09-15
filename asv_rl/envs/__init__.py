"""The waypoint-following task, shared by every algorithm.

Re-exported lazily: the environment imports tf_agents, which callers that only
want configuration or trajectory geometry should not have to pay for.
"""

__all__ = ["ShipEnv", "TRAIN", "EVALUATE"]


def __getattr__(name):
    if name in __all__:
        from asv_rl.envs import ship_env

        return getattr(ship_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
