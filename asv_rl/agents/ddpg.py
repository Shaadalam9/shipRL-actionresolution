"""DDPG agent for the continuous-rudder variant of the task."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import common

from asv_rl.agents.common import LearningRateConfig, resolve_activation


@dataclass
class GaussianNoiseConfig:
    """Zero-mean Gaussian exploration noise with a linearly decaying scale.

    The standard deviation falls from ``std`` to zero over ``decay_episodes``
    and is then held at zero, so late training is effectively greedy.
    """

    std: float = 0.15
    decay_episodes: int = 9000

    def scale(self, episode: int) -> float:
        return float(max(self.std * (1 - episode / self.decay_episodes), 0.0))


@dataclass
class ActorConfig:
    layers: list = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    learning_rate: LearningRateConfig = field(default_factory=lambda: LearningRateConfig(initial=3e-4))


@dataclass
class CriticConfig:
    observation_layers: list = field(default_factory=lambda: [32, 32])
    action_layers: list = field(default_factory=lambda: [16, 16])
    joint_layers: list = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    learning_rate: LearningRateConfig = field(default_factory=lambda: LearningRateConfig(initial=1e-3))


@dataclass
class DdpgAgentConfig:
    """Hyperparameters of the DDPG agent."""

    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    discount: float = 0.95
    target_update_tau: float = 0.01
    target_update_period: int = 1
    exploration: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)


def build_agent(tf_env, config: DdpgAgentConfig, train_step_counter: tf.Variable):
    """Construct a ``DdpgAgent`` for the given environment."""
    actor_net = actor_network.ActorNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=tuple(config.actor.layers),
        activation_fn=resolve_activation(config.actor.activation),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        last_kernel_initializer=tf.keras.initializers.GlorotNormal(),
    )

    critic_net = critic_network.CriticNetwork(
        (tf_env.observation_spec(), tf_env.action_spec()),
        observation_fc_layer_params=tuple(config.critic.observation_layers),
        action_fc_layer_params=tuple(config.critic.action_layers),
        joint_fc_layer_params=tuple(config.critic.joint_layers),
        activation_fn=resolve_activation(config.critic.activation),
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        last_kernel_initializer=tf.keras.initializers.GlorotNormal(),
    )

    agent = ddpg_agent.DdpgAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=config.actor.learning_rate.optimizer(),
        critic_optimizer=config.critic.learning_rate.optimizer(),
        gamma=config.discount,
        target_update_tau=config.target_update_tau,
        target_update_period=config.target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )
    agent.initialize()
    return agent


def make_explorer(config: DdpgAgentConfig, tf_env):
    """Return a function that adds decaying Gaussian noise to an action step.

    The noisy action is clipped back into the rudder limits declared by the
    environment's action spec.
    """
    spec = tf_env.action_spec()
    low = float(np.asarray(spec.minimum).reshape(-1)[0])
    high = float(np.asarray(spec.maximum).reshape(-1)[0])

    def explore(action_step: PolicyStep, episode: int) -> PolicyStep:
        scale = config.exploration.scale(episode)
        action = action_step.action
        if scale > 0:
            noise = np.random.normal(loc=0.0, scale=scale, size=action.shape)
            action = action + tf.constant(noise, dtype=action.dtype)
        return PolicyStep(tf.clip_by_value(action, low, high))

    return explore
