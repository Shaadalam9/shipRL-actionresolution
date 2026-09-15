"""Deep Q-Network agent for the discrete-rudder variants of the task.

The number of rudder settings is a property of the environment
(``environment.action_space.rudder_angles_deg``), so the same agent code backs
both the 3-action and the 5-action experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import common

from asv_rl.agents.common import LearningRateConfig, NetworkConfig


@dataclass
class EpsilonGreedyConfig:
    """Linearly annealed epsilon-greedy exploration.

    Epsilon falls from 1.0 to ``final_epsilon`` over ``decay_episodes``
    episodes and stays there. A non-zero floor keeps a trickle of exploration
    alive for the whole run.
    """

    decay_episodes: int = 5000
    final_epsilon: float = 0.0

    def epsilon(self, episode: int) -> float:
        return float(np.maximum(1 - (episode / self.decay_episodes), self.final_epsilon))


@dataclass
class DqnAgentConfig:
    """Hyperparameters of the DQN agent."""

    q_network: NetworkConfig = field(default_factory=lambda: NetworkConfig(layers=[64, 64], activation="tanh"))
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    discount: float = 0.95
    target_update_tau: float = 0.01
    target_update_period: int = 1
    exploration: EpsilonGreedyConfig = field(default_factory=EpsilonGreedyConfig)


def build_agent(tf_env, config: DqnAgentConfig, train_step_counter: tf.Variable):
    """Construct a ``DqnAgent`` for the given environment."""
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=config.q_network.fc_layer_params,
        activation_fn=config.q_network.activation_fn,
    )

    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=config.learning_rate.optimizer(),
        gamma=config.discount,
        target_update_tau=config.target_update_tau,
        target_update_period=config.target_update_period,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )
    agent.initialize()
    return agent


def make_explorer(config: DqnAgentConfig, tf_env):
    """Return a function that injects epsilon-greedy noise into an action step.

    The number of actions is read from the environment's action spec, so a
    config that changes the rudder set needs no change here.
    """
    num_actions = int(tf_env.action_spec().maximum - tf_env.action_spec().minimum + 1)

    def explore(action_step: PolicyStep, episode: int) -> PolicyStep:
        if np.random.random() < config.exploration.epsilon(episode):
            action = tf.constant(
                [np.random.randint(0, num_actions)], shape=(1,), dtype=np.int64, name="action"
            )
            return PolicyStep(action)
        return action_step

    return explore
