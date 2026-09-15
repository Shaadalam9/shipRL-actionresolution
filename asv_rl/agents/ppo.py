"""PPO (clipped surrogate) agent for the continuous-rudder variant of the task.

PPO is on-policy, so it has no replay buffer to warm up and no exploration
wrapper: the collect policy is itself stochastic. Training proceeds in
iterations, each of which collects a fresh batch of episodes and then performs
``epochs`` passes over that batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network, value_network

from asv_rl.agents.common import LearningRateConfig, NetworkConfig


@dataclass
class PpoAgentConfig:
    """Hyperparameters of the PPO agent."""

    actor: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(layers=[128, 128], activation="tanh")
    )
    value: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(layers=[128, 128], activation="tanh")
    )
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    discount: float = 0.95
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    entropy_regularization: float = 0.01
    epochs: int = 10
    greedy_eval: bool = True
    normalize_rewards: bool = True
    normalize_observations: bool = True


def build_agent(tf_env, config: PpoAgentConfig, train_step_counter: tf.Variable):
    """Construct a ``PPOClipAgent`` for the given environment."""
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=config.actor.fc_layer_params,
        activation_fn=config.actor.activation_fn,
    )

    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=config.value.fc_layer_params,
        activation_fn=config.value.activation_fn,
    )

    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        optimizer=config.learning_rate.optimizer(),
        actor_net=actor_net,
        value_net=value_net,
        greedy_eval=config.greedy_eval,
        importance_ratio_clipping=config.clip_ratio,
        lambda_value=config.gae_lambda,
        discount_factor=config.discount,
        entropy_regularization=config.entropy_regularization,
        num_epochs=config.epochs,
        use_gae=True,
        use_td_lambda_return=True,
        normalize_rewards=config.normalize_rewards,
        normalize_observations=config.normalize_observations,
        debug_summaries=False,
        train_step_counter=train_step_counter,
    )
    agent.initialize()
    return agent
