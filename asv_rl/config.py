"""Typed configuration objects loaded from YAML experiment files.

A single YAML file fully describes one training run: the environment, the
agent, and the training schedule. ``load_config`` turns such a file into the
nested dataclasses below, rejecting unknown keys so that a misspelled
hyperparameter fails loudly instead of being silently ignored.

The algorithm-specific ``agent`` and ``training`` sections are built by the
algorithm's own module; see :mod:`asv_rl.agents.registry`.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or incomplete."""


# ---------------------------------------------------------------------------
# Generic dataclass construction
# ---------------------------------------------------------------------------

def _is_optional(annotation) -> bool:
    return typing.get_origin(annotation) is typing.Union and type(None) in typing.get_args(annotation)


def _unwrap_optional(annotation):
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    return args[0] if len(args) == 1 else annotation


def build(cls, data: Mapping[str, Any] | None, path: str = ""):
    """Recursively build dataclass ``cls`` from a mapping.

    Unknown keys raise :class:`ConfigError` naming the offending path, which is
    the single most useful thing a config loader can do for a new user.
    """
    data = {} if data is None else data
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(
            f"{path or 'config'}: unknown option(s) {sorted(unknown)}. "
            f"Valid options are {sorted(fields)}."
        )

    # ``from __future__ import annotations`` turns field.type into a string, so
    # resolve the real types before looking for nested dataclasses.
    hints = typing.get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        annotation = hints.get(name, f.type)
        if _is_optional(annotation):
            annotation = _unwrap_optional(annotation)
        child_path = f"{path}.{name}" if path else name

        if name not in data:
            has_default = f.default is not dataclasses.MISSING
            has_factory = f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            if not (has_default or has_factory):
                raise ConfigError(f"{child_path}: required option is missing")
            continue

        value = data[name]
        if dataclasses.is_dataclass(annotation) and value is not None:
            kwargs[name] = build(annotation, value, child_path)
        else:
            kwargs[name] = value

    return cls(**kwargs)


def to_dict(obj) -> Any:
    """Convert nested dataclasses back to plain dicts for YAML serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class WindConfig:
    """Steady wind loading. ``speed`` is non-dimensional, direction in degrees."""

    enabled: bool = False
    speed: float = 0.0
    direction_deg: float = 0.0


@dataclass
class WaveConfig:
    """Mean wave drift loading, interpolated from the shipped KCS drift table."""

    enabled: bool = False
    height: float = 0.0
    period: float = 0.0
    direction_deg: float = 0.0


@dataclass
class ConditionsConfig:
    """Environmental disturbances acting on the hull."""

    wind: WindConfig = field(default_factory=WindConfig)
    wave: WaveConfig = field(default_factory=WaveConfig)


@dataclass
class ActionSpaceConfig:
    """Rudder command space.

    ``discrete`` exposes the fixed rudder angles in ``rudder_angles_deg`` as a
    categorical action (used by the DQN variants); ``continuous`` exposes a
    single bounded rudder angle in +/- ``max_rudder_deg`` (DDPG and PPO).
    """

    type: str = "continuous"
    rudder_angles_deg: Sequence[float] = field(default_factory=list)
    max_rudder_deg: float = 35.0

    def __post_init__(self):
        if self.type not in ("discrete", "continuous"):
            raise ConfigError(
                f"environment.action_space.type: expected 'discrete' or 'continuous', got {self.type!r}"
            )
        if self.type == "discrete" and not self.rudder_angles_deg:
            raise ConfigError(
                "environment.action_space.rudder_angles_deg: required when type is 'discrete'"
            )


@dataclass
class RewardConfig:
    """Weights of the three-term shaping reward.

    The total reward per step is::

        cross_track_scale * exp(-cross_track_decay * xte^2) + cross_track_offset
      + course_scale      * exp(-course_decay * |course_error|) + course_offset
      - distance_weight   * distance_to_goal
    """

    cross_track_scale: float = 2.0
    cross_track_decay: float = 0.08
    cross_track_offset: float = -1.0
    course_scale: float = 1.3
    course_decay: float = 10.0
    course_offset: float = -0.3
    distance_weight: float = 0.25


@dataclass
class EnvironmentConfig:
    """Everything about the path-following task itself."""

    episode_steps: int = 160
    integration_dt: float = 0.3
    arrival_radius: float = 1.5
    terminal_reward: float = 100.0
    initial_surge: float = 1.0
    goal_radius_range: Sequence[float] = (8, 28)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)


@dataclass
class ReplayBufferConfig:
    """Uniform replay buffer settings shared by the off-policy agents."""

    capacity: int = 100_000
    batch_size: int = 128
    n_step: int = 2
    parallel_calls: int = 2
    prefetch: int = 3


@dataclass
class OffPolicyTrainingConfig:
    """Training schedule for DQN and DDPG."""

    episodes: int = 10_001
    warmup_episodes: int = 10
    update_every_steps: int = 10
    checkpoint_every: int = 1000
    metric_window: int = 100
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)


@dataclass
class OnPolicyTrainingConfig:
    """Training schedule for PPO.

    ``checkpoint_after`` suppresses early checkpoints, which are rarely worth
    the disk space; set it to 0 to save from the first eligible iteration.
    """

    iterations: int = 100
    episodes_per_iteration: int = 50
    checkpoint_every: int = 5
    checkpoint_after: int = 50
    metric_window: int = 1


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """A fully resolved experiment: environment + agent + training schedule."""

    algorithm: str
    experiment: str
    run_name: str
    seed: int
    environment: EnvironmentConfig
    agent: Any
    training: Any
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "experiment": self.experiment,
            "run_name": self.run_name,
            "seed": self.seed,
            "description": self.description,
            "environment": to_dict(self.environment),
            "agent": to_dict(self.agent),
            "training": to_dict(self.training),
        }

    def save(self, path: str | Path) -> None:
        """Write the resolved config next to the artefacts it produced."""
        with open(path, "w") as fid:
            yaml.safe_dump(self.as_dict(), fid, sort_keys=False, default_flow_style=False)


_TOP_LEVEL_KEYS = {
    "algorithm",
    "experiment",
    "run_name",
    "seed",
    "description",
    "environment",
    "agent",
    "training",
}


def apply_overrides(data: dict, overrides: Sequence[str]) -> dict:
    """Apply ``dotted.key=value`` overrides to a raw config mapping.

    Values are parsed as YAML, so ``training.episodes=500``, ``agent.discount=0.99``
    and ``environment.conditions.wind.enabled=true`` all do the expected thing.
    This is what replaces the old ``batch_train.py`` parameter sweeps.
    """
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"--set {override!r}: expected the form key.path=value")
        dotted, _, raw_value = override.partition("=")
        keys = dotted.strip().split(".")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {override!r}: could not parse value: {exc}") from exc

        node = data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise ConfigError(f"--set {override!r}: {key!r} is not a section of this config")
            node = node[key]
        if keys[-1] not in node:
            raise ConfigError(
                f"--set {override!r}: {dotted!r} is not defined in the config; "
                "overrides may only change existing options"
            )
        node[keys[-1]] = value
    return data


def load_raw(path: str | Path) -> dict:
    """Read a YAML config file and validate its top-level shape."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as fid:
        data = yaml.safe_load(fid)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")

    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    for required in ("algorithm", "environment", "agent", "training"):
        if required not in data:
            raise ConfigError(f"{path}: missing required top-level key {required!r}")
    return data
