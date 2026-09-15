"""Waypoint-following environment for the KCS hull.

One environment serves every algorithm in this repository. Whether the agent
commands a small set of fixed rudder angles (DQN) or a continuous rudder angle
(DDPG, PPO) is decided by ``config.action_space``; everything else — dynamics,
observation, reward, termination — is shared, so a comparison between
algorithms is a comparison of the algorithms and not of four slightly
divergent copies of the task.

The observation is 4-dimensional::

    [cross-track error, course angle error, distance to goal, yaw rate]

The environment runs in one of two modes. ``train`` samples a single random
goal per episode. ``evaluate`` follows a supplied list of waypoints in order
and records the full trajectory for plotting and error metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from asv_rl.config import EnvironmentConfig
from asv_rl.dynamics import kcs

#: Bounds of the observation vector, used to build the observation spec.
OBSERVATION_MINIMUM = [-20.0, -np.pi, 0.0, -1.0]
OBSERVATION_MAXIMUM = [20.0, np.pi, 50.0, 1.0]

TRAIN = "train"
EVALUATE = "evaluate"


class ShipEnv(py_environment.PyEnvironment):
    """A ``tf_agents`` environment wrapping the 3-DOF KCS maneuvering model.

    Args:
        config: Task definition (action space, reward weights, disturbances).
        mode: ``"train"`` for random single-goal episodes, ``"evaluate"`` to
            track a supplied waypoint list.
        x_waypoints, y_waypoints: Waypoint coordinates, required in evaluate mode.
        initial_state: Initial non-dimensional state vector
            ``[u, v, r, x, y, psi, delta]``, required in evaluate mode.
        verbose: Print per-waypoint progress. Off by default because training
            runs for thousands of episodes.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        mode: str = TRAIN,
        x_waypoints: Sequence[float] | None = None,
        y_waypoints: Sequence[float] | None = None,
        initial_state: Sequence[float] | None = None,
        verbose: bool = False,
    ):
        if mode not in (TRAIN, EVALUATE):
            raise ValueError(f"mode must be {TRAIN!r} or {EVALUATE!r}, got {mode!r}")
        if mode == EVALUATE:
            if x_waypoints is None or y_waypoints is None:
                raise ValueError("evaluate mode requires x_waypoints and y_waypoints")
            if len(x_waypoints) != len(y_waypoints):
                raise ValueError(
                    f"waypoint lists differ in length: {len(x_waypoints)} x vs {len(y_waypoints)} y"
                )
            if len(x_waypoints) < 2:
                raise ValueError("evaluate mode needs at least a start and a goal waypoint")
            if initial_state is None:
                raise ValueError("evaluate mode requires initial_state")

        self.config = config
        self.mode = mode
        self.verbose = verbose

        action_config = config.action_space
        if action_config.type == "discrete":
            self._rudder_angles = np.asarray(action_config.rudder_angles_deg, dtype=np.float64) * np.pi / 180
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int64, minimum=0, maximum=len(self._rudder_angles) - 1, name="action"
            )
        else:
            self._rudder_angles = None
            limit = action_config.max_rudder_deg * np.pi / 180
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.float32, minimum=-limit, maximum=limit, name="action"
            )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,),
            dtype=np.float32,
            minimum=OBSERVATION_MINIMUM,
            maximum=OBSERVATION_MAXIMUM,
            name="observation",
        )

        self.test_x_waypoints = list(x_waypoints) if x_waypoints is not None else None
        self.test_y_waypoints = list(y_waypoints) if y_waypoints is not None else None
        self.nwp = len(x_waypoints) if x_waypoints is not None else None
        # Copied so that a caller reusing the same list across runs is not
        # mutated by the integrator writing back into the state vector.
        self.initial_state = list(initial_state) if initial_state is not None else None

        self.episode_ended = False
        self.counter = 0
        self.wp_counter = 1
        self.x_goal = 0.0
        self.y_goal = 0.0
        self.distance = 0.0
        self.state = [config.initial_surge, 0, 0, 0, 0, 0, 0]

        self._reset_trajectories()

    # -- specs -------------------------------------------------------------

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # -- trajectory recording ---------------------------------------------

    def _reset_trajectories(self):
        """Clear the recorded trajectories. Populated in evaluate mode only."""
        self.x_traj, self.y_traj, self.psi_traj = [], [], []
        self.u_traj, self.v_traj, self.r_traj = [], [], []
        self.delta_traj = []
        self.action_traj = []
        self.cross_trk_err_traj, self.course_ang_err_traj, self.distance_traj = [], [], []
        self.total_reward, self.reward_cross_track, self.reward_course, self.reward_distance = [], [], [], []

    # -- helpers -----------------------------------------------------------

    def _rudder_command(self, action) -> float:
        """Map an agent action to a commanded rudder angle in radians."""
        if self._rudder_angles is not None:
            index = int(np.asarray(action).reshape(-1)[0])
            if not 0 <= index < len(self._rudder_angles):
                raise ValueError(
                    f"discrete action {index} out of range for "
                    f"{len(self._rudder_angles)} rudder settings"
                )
            return float(self._rudder_angles[index])
        return float(np.asarray(action).reshape(-1)[0])

    def _integrate(self, delta_c: float):
        """Advance the hull dynamics by one control interval."""
        conditions = self.config.conditions
        tspan = (0, self.config.integration_dt)
        sol = solve_ivp(
            lambda t, v: kcs.KCS_ode(
                t,
                v,
                delta_c,
                wind_flag=int(conditions.wind.enabled),
                wind_speed=conditions.wind.speed,
                wind_dir=conditions.wind.direction_deg,
                wave_flag=int(conditions.wave.enabled),
                wave_height=conditions.wave.height,
                wave_period=conditions.wave.period,
                wave_dir=conditions.wave.direction_deg,
            ),
            tspan,
            self.state,
            t_eval=tspan,
            dense_output=True,
        )
        return [row[-1] for row in sol.y]

    def _reward(self, cross_track_error, course_angle_err, distance):
        """Three-term shaping reward; see :class:`~asv_rl.config.RewardConfig`."""
        weights = self.config.reward
        r_cross = weights.cross_track_scale * np.exp(
            -weights.cross_track_decay * cross_track_error ** 2
        ) + weights.cross_track_offset
        r_course = weights.course_scale * np.exp(
            -weights.course_decay * abs(course_angle_err)
        ) + weights.course_offset
        r_distance = -distance * weights.distance_weight
        return r_cross, r_course, r_distance

    # -- environment API ---------------------------------------------------

    def _step(self, action):
        delta_c = self._rudder_command(action)
        u, v, r, x, y, psi_rad, delta = self._integrate(delta_c)
        psi = psi_rad % (2 * np.pi)

        self.state = [u, v, r, x, y, psi, delta]

        recording = self.mode == EVALUATE
        if recording:
            self.action_traj.append(delta_c)
            self.x_traj.append(x)
            self.y_traj.append(y)
            self.psi_traj.append(psi_rad)
            self.u_traj.append(u)
            self.v_traj.append(v)
            self.r_traj.append(r)
            self.delta_traj.append(delta)

        if recording:
            x_init = self.test_x_waypoints[self.wp_counter - 1]
            y_init = self.test_y_waypoints[self.wp_counter - 1]
            x_goal = self.test_x_waypoints[self.wp_counter]
            y_goal = self.test_y_waypoints[self.wp_counter]
        else:
            x_init, y_init = 0.0, 0.0
            x_goal, y_goal = self.x_goal, self.y_goal

        # Distance to the active goal
        distance = ((x - x_goal) ** 2 + (y - y_goal) ** 2) ** 0.5
        self.distance = distance

        # Cross-track error: signed offset from the leg being tracked.
        # Written out rather than via np.cross, which dropped support for
        # 2-D vectors in NumPy 2.0.
        vec1 = np.array([x_goal - x_init, y_goal - y_init])
        vec2 = np.array([x_goal - x, y_goal - y])
        vec1_hat = vec1 / np.linalg.norm(vec1)
        cross_track_error = vec2[0] * vec1_hat[1] - vec2[1] * vec1_hat[0]

        # Course angle error: heading of the velocity vector vs. bearing to goal
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        Uvec = np.array([x_dot, y_dot])
        Uvec_hat = Uvec / np.linalg.norm(Uvec)
        vec2_hat = vec2 / np.linalg.norm(vec2)

        course_angle = np.arctan2(Uvec[1], Uvec[0])
        psi_vec2 = np.arctan2(vec2[1], vec2[0])
        course_angle_err = (course_angle - psi_vec2 + np.pi) % (2 * np.pi) - np.pi

        r_cross, r_course, r_distance = self._reward(cross_track_error, course_angle_err, distance)
        reward = r_cross + r_course + r_distance

        if recording:
            self.distance_traj.append(distance)
            self.cross_trk_err_traj.append(cross_track_error)
            self.course_ang_err_traj.append(course_angle_err)
            self.reward_cross_track.append(r_cross)
            self.reward_course.append(r_course)
            self.reward_distance.append(r_distance)
            self.total_reward.append(reward)

        observation = np.array(
            [cross_track_error, course_angle_err, distance, r], dtype=np.float32
        )

        # Waypoint reached
        if abs(distance) <= self.config.arrival_radius:
            if recording:
                if self.verbose:
                    print(f"Waypoint {self.wp_counter} reached")
                if self.wp_counter == self.nwp - 1:
                    self.episode_ended = True
                    return ts.termination(observation, reward)
                self.wp_counter += 1
            else:
                self.episode_ended = True
                return ts.termination(observation, self.config.terminal_reward)

        # Overshoot: the ship has passed the goal and is heading away from it
        angle_btw23 = np.arccos(np.clip(np.dot(vec2_hat, Uvec_hat), -1.0, 1.0))
        angle_btw12 = np.arccos(np.clip(np.dot(vec1_hat, vec2_hat), -1.0, 1.0))
        if angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2:
            self.episode_ended = True
            if recording and self.verbose:
                print(f"Waypoint {self.wp_counter} missed")
            return ts.termination(observation, reward)

        return ts.transition(observation, reward)

    def _reset(self):
        self._reset_trajectories()
        self.episode_ended = False
        self.wp_counter = 1

        if self.mode == TRAIN:
            self.state = [self.config.initial_surge, 0, 0, 0, 0, 0, 0]

            low, high = self.config.goal_radius_range
            radius = np.random.randint(int(low), int(high))
            random_theta = 2 * np.pi * np.random.random()
            self.x_goal = radius * np.cos(random_theta)
            self.y_goal = radius * np.sin(random_theta)
            self.counter += 1

            x, y = 0.0, 0.0
            x_dot, y_dot = 1.0, 0.0
            distance = radius
        else:
            self.state = list(self.initial_state)
            self.x_goal = self.test_x_waypoints[1]
            self.y_goal = self.test_y_waypoints[1]

            x = self.test_x_waypoints[0]
            y = self.test_y_waypoints[0]
            u, v, psi = self.state[0], self.state[1], self.state[5]
            x_dot = u * np.cos(psi) - v * np.sin(psi)
            y_dot = u * np.sin(psi) + v * np.cos(psi)
            distance = np.sqrt((self.x_goal - x) ** 2 + (self.y_goal - y) ** 2)

        if self.verbose:
            print(f"Goal coordinates: ({self.x_goal:.3f}, {self.y_goal:.3f})")

        course_angle = np.arctan2(y_dot, x_dot)
        psi_vec2 = np.arctan2(self.y_goal - y, self.x_goal - x)
        course_angle_err = (course_angle - psi_vec2 + np.pi) % (2 * np.pi) - np.pi

        observation = np.array([0, course_angle_err, distance, 0], dtype=np.float32)
        return ts.restart(observation)

    # -- metrics -----------------------------------------------------------

    def cross_track_rmse(self) -> float:
        """Root-mean-square cross-track error over the recorded trajectory."""
        if not self.cross_trk_err_traj:
            raise RuntimeError("no trajectory recorded; run the environment in evaluate mode")
        return float(np.sqrt(np.mean(np.square(self.cross_trk_err_traj))))

    def mean_absolute_rudder(self) -> float:
        """Mean absolute realised rudder angle, a proxy for actuator effort."""
        if not self.delta_traj:
            raise RuntimeError("no trajectory recorded; run the environment in evaluate mode")
        return float(np.mean(np.abs(self.delta_traj)))
