import gym
import numpy as np
from gym import spaces
from env.arm_env import ArmEnv


class GymArmEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False):
        super().__init__()

        # -----------------------
        # Low-level environment
        # -----------------------
        self.env = ArmEnv(gui=gui)

        # -----------------------
        # Action & Observation
        # -----------------------
        # Action: delta end-effector (x, y, z)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation: joint(7) + ee(3) + obj(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # -----------------------
        # Episode params
        # -----------------------
        self.max_steps = 100
        self.step_count = 0

        # -----------------------
        # Internal state
        # -----------------------
        self.joint_positions = np.zeros(7)
        self.ee_target = None

        self.prev_distance = None
        self.prev_ee_pos = None

    # =========================================================
    # Gym API
    # =========================================================

    def reset(self):
        obs = self.env.reset()

        self.joint_positions = obs[:7].copy()
        self.ee_target = obs[7:10].copy()

        self.prev_distance = self.env.compute_distance()
        self.prev_ee_pos = obs[7:10].copy()

        self.step_count = 0
        return obs.astype(np.float32)

    def step(self, action):
        self.step_count += 1

        self._apply_action(action)
        obs = self._simulate()
        reward = self._compute_reward(obs)
        done, success = self._check_done()

        info = {
            "distance": self.prev_distance,
            "success": success
        }

        return obs.astype(np.float32), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    # =========================================================
    # Internal helpers
    # =========================================================

    def _apply_action(self, action):
        """Convert RL action → EE target update"""
        action = np.clip(action, -1.0, 1.0)
        ee_delta = action * 0.01  # meters per step

        self.ee_target += ee_delta

        # Workspace limits (VERY important)
        self.ee_target[0] = np.clip(self.ee_target[0], 0.3, 0.8)
        self.ee_target[1] = np.clip(self.ee_target[1], -0.4, 0.4)
        self.ee_target[2] = np.clip(self.ee_target[2], 0.03, 0.6)

    def _simulate(self):
        """Run IK + physics"""
        obs = self.env.step_ik(self.ee_target, self.joint_positions)
        self.joint_positions = obs[:7].copy()
        return obs

    # =========================================================
    # Reward functions
    # =========================================================

    def _compute_reward(self, obs):
        dist = self.env.compute_distance()

        distance_reward = self._distance_reward(dist)
        directional_reward = self._directional_reward(obs)
        z_reward = self._z_reward(obs, dist)
        smoothness_penalty = self._smoothness_penalty()

        reward = (
            distance_reward
            + 0.3 * directional_reward
            + z_reward
            - smoothness_penalty
            + 0.01
        )

        self.prev_distance = dist
        return reward

    def _distance_reward(self, dist):
        scale = 10.0 if dist > 0.15 else 5.0
        reward = scale * (self.prev_distance - dist)

        if dist < 0.15:
            reward += 1.0
        if dist < 0.10:
            reward += 2.0
        if dist < 0.07:
            reward += 3.0

        return reward

    def _directional_reward(self, obs):
        ee_pos = obs[7:10]
        obj_pos = obs[10:13]

        ee_vel = ee_pos - self.prev_ee_pos
        self.prev_ee_pos = ee_pos.copy()

        direction = obj_pos - ee_pos
        direction_unit = direction / (np.linalg.norm(direction) + 1e-6)

        return np.dot(ee_vel, direction_unit)

    def _z_reward(self, obs, dist):
        ee_z = obs[9]
        obj_z = obs[12]

        z_error = ee_z - obj_z
        reward = -2.0 * abs(z_error)

        # Encourage downward motion when close
        if dist < 0.25:
            reward += 1.5 * (self.prev_ee_pos[2] - ee_z)

        return reward

    def _smoothness_penalty(self):
        return 0.02 * np.linalg.norm(self.ee_target - self.prev_ee_pos)

    # =========================================================
    # Termination
    # =========================================================

    def _check_done(self):
        success = False
        done = False

        if self.prev_distance < 0.05:
            success = True
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return done, success
