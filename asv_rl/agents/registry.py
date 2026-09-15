"""Mapping from algorithm name to its config classes, builder, and trainer.

Adding an algorithm means writing a module beside this one and appending a
single :class:`AlgorithmSpec` entry to ``ALGORITHMS`` — nothing else in the
codebase needs to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from asv_rl.config import (
    ConfigError,
    ExperimentConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    build,
    load_raw,
)

OFF_POLICY = "off_policy"
ON_POLICY = "on_policy"


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the training and evaluation entry points need for one algorithm."""

    name: str
    agent_config_cls: type
    training_config_cls: type
    build_agent: Callable[..., Any]
    trainer: str
    make_explorer: Callable[..., Any] | None = None
    action_space_type: str = "continuous"
    description: str = ""


def _specs() -> dict[str, AlgorithmSpec]:
    # Imported lazily so that `--help` and config validation do not pay the
    # cost of importing TensorFlow.
    from asv_rl.agents import ddpg, dqn, ppo

    return {
        "dqn": AlgorithmSpec(
            name="dqn",
            agent_config_cls=dqn.DqnAgentConfig,
            training_config_cls=OffPolicyTrainingConfig,
            build_agent=dqn.build_agent,
            make_explorer=dqn.make_explorer,
            trainer=OFF_POLICY,
            action_space_type="discrete",
            description="Deep Q-Network over a fixed set of rudder angles",
        ),
        "ddpg": AlgorithmSpec(
            name="ddpg",
            agent_config_cls=ddpg.DdpgAgentConfig,
            training_config_cls=OffPolicyTrainingConfig,
            build_agent=ddpg.build_agent,
            make_explorer=ddpg.make_explorer,
            trainer=OFF_POLICY,
            action_space_type="continuous",
            description="Deep Deterministic Policy Gradient with a continuous rudder",
        ),
        "ppo": AlgorithmSpec(
            name="ppo",
            agent_config_cls=ppo.PpoAgentConfig,
            training_config_cls=OnPolicyTrainingConfig,
            build_agent=ppo.build_agent,
            make_explorer=None,
            trainer=ON_POLICY,
            action_space_type="continuous",
            description="Proximal Policy Optimization with a clipped surrogate objective",
        ),
    }


#: Algorithm names accepted in the ``algorithm`` field of a config file.
ALGORITHM_NAMES = ("dqn", "ddpg", "ppo")


def get_spec(name: str) -> AlgorithmSpec:
    """Look up an algorithm by the name used in a config file."""
    specs = _specs()
    try:
        return specs[name]
    except KeyError:
        raise ConfigError(
            f"unknown algorithm {name!r}; available: {sorted(specs)}"
        ) from None


def load_config(path, overrides=()) -> tuple[ExperimentConfig, AlgorithmSpec]:
    """Load and fully validate an experiment config file.

    Returns the resolved config together with the spec of its algorithm, so
    callers never have to re-derive one from the other.
    """
    from asv_rl.config import EnvironmentConfig, apply_overrides

    raw = load_raw(path)
    raw = apply_overrides(raw, list(overrides))

    spec = get_spec(raw["algorithm"])
    environment = build(EnvironmentConfig, raw["environment"], "environment")

    if environment.action_space.type != spec.action_space_type:
        raise ConfigError(
            f"algorithm {spec.name!r} needs a {spec.action_space_type} action space, "
            f"but environment.action_space.type is {environment.action_space.type!r}"
        )

    # `experiment` groups runs that share a config (the two DQN action sets are
    # separate experiments of one algorithm), and names the model subdirectory.
    config = ExperimentConfig(
        algorithm=spec.name,
        experiment=raw.get("experiment", spec.name),
        run_name=raw.get("run_name", "default"),
        seed=int(raw.get("seed", 0)),
        description=raw.get("description", ""),
        environment=environment,
        agent=build(spec.agent_config_cls, raw["agent"], "agent"),
        training=build(spec.training_config_cls, raw["training"], "training"),
    )
    return config, spec
