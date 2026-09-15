"""Configuration fragments and helpers shared by every agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import tensorflow as tf

from asv_rl.config import ConfigError

#: Activation functions selectable from a config file.
ACTIVATIONS = {
    "relu": tf.keras.activations.relu,
    "tanh": tf.keras.activations.tanh,
    "elu": tf.keras.activations.elu,
    "sigmoid": tf.keras.activations.sigmoid,
}


def resolve_activation(name: str):
    """Look up a Keras activation by name, with a helpful error on typos."""
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ConfigError(
            f"unknown activation {name!r}; available: {sorted(ACTIVATIONS)}"
        ) from None


@dataclass
class LearningRateConfig:
    """An Adam learning rate, optionally on an exponential decay schedule.

    Decay is applied only when both ``decay_steps`` and ``decay_rate`` are set;
    leaving either unset gives a constant learning rate.
    """

    initial: float = 1e-3
    decay_steps: int | None = None
    decay_rate: float | None = None

    def build(self):
        """Return either a float or a Keras schedule, as Adam accepts both."""
        if self.decay_steps is None or self.decay_rate is None:
            return self.initial
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
        )

    def optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.build())


@dataclass
class NetworkConfig:
    """A plain fully-connected network."""

    layers: Sequence[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"

    @property
    def fc_layer_params(self) -> tuple:
        return tuple(self.layers)

    @property
    def activation_fn(self):
        return resolve_activation(self.activation)
