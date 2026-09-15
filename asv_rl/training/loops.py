"""Training loops shared by the algorithms in this repository.

There are exactly two: an off-policy loop with a replay buffer (DQN, DDPG) and
an on-policy loop that rebuilds its batch every iteration (PPO). Both write the
same artefacts, so runs remain directly comparable.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment, wrappers
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from asv_rl.envs.ship_env import TRAIN, ShipEnv
from asv_rl.training.artifacts import RunArtifacts

TERMINAL_STEP_TYPE = 2


def build_training_env(config):
    """Wrap the task in a time limit and hand it to ``tf_agents``."""
    env = wrappers.TimeLimit(
        ShipEnv(config.environment, mode=TRAIN),
        duration=config.environment.episode_steps,
    )
    return tf_py_environment.TFPyEnvironment(env)


def collect_episode(tf_env, policy, buffer, explore=None, episode=0, on_step=None) -> float:
    """Run one episode, push every transition into ``buffer``, return its return.

    Args:
        explore: Optional ``(action_step, episode) -> action_step`` hook that
            injects exploration noise.
        on_step: Optional callback invoked once per environment step, used by
            the off-policy loop to interleave gradient updates with collection.
    """
    time_step = tf_env.reset()
    episode_return = 0.0

    while not np.equal(time_step.step_type, TERMINAL_STEP_TYPE):
        action_step = policy.action(time_step)
        if explore is not None:
            action_step = explore(action_step, episode)

        next_time_step = tf_env.step(action_step.action)
        episode_return += float(next_time_step.reward.numpy()[0])
        buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))
        time_step = next_time_step

        if on_step is not None:
            on_step()

    return episode_return


def _progress(prefix, index, total, value_name, value, started_at):
    elapsed = time.time() - started_at
    print(
        f"{prefix} {index}/{total} | {value_name}={value:.3f} | elapsed={elapsed / 60:.1f} min",
        flush=True,
    )


def train_off_policy(config, spec, output_dir) -> dict:
    """Replay-buffer training loop used by DQN and DDPG."""
    training = config.training
    buffer_config = training.replay_buffer

    tf_env = build_training_env(config)
    train_step_counter = tf.Variable(0)
    agent = spec.build_agent(tf_env, config.agent, train_step_counter)
    explore = spec.make_explorer(config.agent, tf_env) if spec.make_explorer else None

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=buffer_config.capacity,
    )

    # Seed the buffer with random experience so the first updates have data.
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    print(f"Warming up replay buffer with {training.warmup_episodes} random episodes...", flush=True)
    for _ in range(training.warmup_episodes):
        collect_episode(tf_env, random_policy, replay_buffer)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=buffer_config.parallel_calls,
        sample_batch_size=buffer_config.batch_size,
        num_steps=buffer_config.n_step,
    ).prefetch(buffer_config.prefetch)
    # Created once and reused: rebuilding the iterator on every update is
    # needlessly expensive and samples no differently.
    iterator = iter(dataset)

    agent.train_step_counter.assign(0)
    artifacts = RunArtifacts(output_dir, config)

    returns, losses = [], []
    step_counter = 0
    started_at = time.time()

    def update_if_due():
        nonlocal step_counter
        if step_counter % training.update_every_steps == 0:
            experience, _ = next(iterator)
            agent.train(experience)
        step_counter += 1

    for episode in range(training.episodes):
        episode_return = collect_episode(
            tf_env,
            agent.policy,
            replay_buffer,
            explore=explore,
            episode=episode,
            on_step=update_if_due,
        )
        returns.append(episode_return)

        experience, _ = next(iterator)
        loss = float(agent.train(experience).loss)
        losses.append(loss)

        if (episode + 1) % 10 == 0 or episode == 0:
            window = min(len(returns), training.metric_window)
            _progress(
                "episode",
                episode + 1,
                training.episodes,
                f"return(avg{window})",
                float(np.mean(returns[-window:])),
                started_at,
            )

        if (episode + 1) % training.checkpoint_every == 0:
            path = artifacts.save_checkpoint(agent, episode + 1)
            print(f"  saved checkpoint -> {path}", flush=True)

    artifacts.save_final_policy(agent)
    artifacts.save_history(returns, losses, training.metric_window)
    artifacts.save_curves(returns, losses, training.metric_window)
    metrics = artifacts.save_metrics(
        returns, losses, training.metric_window,
        extra={"wall_clock_minutes": round((time.time() - started_at) / 60, 2)},
    )
    return metrics


def train_on_policy(config, spec, output_dir) -> dict:
    """Iteration-based training loop used by PPO.

    Each iteration collects a fresh batch of episodes with the current policy,
    performs one ``agent.train`` call over that batch (which internally runs
    ``agent.epochs`` passes), and then discards the batch.
    """
    training = config.training

    tf_env = build_training_env(config)
    train_step_counter = tf.Variable(0)
    agent = spec.build_agent(tf_env, config.agent, train_step_counter)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=training.episodes_per_iteration * config.environment.episode_steps,
    )

    artifacts = RunArtifacts(output_dir, config)
    episode_returns, iteration_returns, losses = [], [], []
    started_at = time.time()

    for iteration in range(training.iterations):
        replay_buffer.clear()
        batch_returns = []

        for _ in range(training.episodes_per_iteration):
            episode_return = collect_episode(tf_env, agent.collect_policy, replay_buffer)
            batch_returns.append(episode_return)
            episode_returns.append(episode_return)

        experience = replay_buffer.gather_all()
        loss = float(agent.train(experience).loss)
        losses.append(loss)
        iteration_returns.append(float(np.mean(batch_returns)))

        _progress(
            "iteration",
            iteration + 1,
            training.iterations,
            "return(mean)",
            iteration_returns[-1],
            started_at,
        )

        due = (iteration + 1) % training.checkpoint_every == 0
        if due and (iteration + 1) >= training.checkpoint_after:
            path = artifacts.save_checkpoint(agent, iteration + 1)
            print(f"  saved checkpoint -> {path}", flush=True)

    artifacts.save_final_policy(agent)
    artifacts.save_history(iteration_returns, losses, training.metric_window)
    artifacts.save_curves(iteration_returns, losses, training.metric_window)
    metrics = artifacts.save_metrics(
        iteration_returns, losses, training.metric_window,
        extra={
            "episodes": len(episode_returns),
            "iterations": training.iterations,
            "wall_clock_minutes": round((time.time() - started_at) / 60, 2),
        },
    )
    return metrics
