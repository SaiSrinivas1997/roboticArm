import gym
import numpy as np
from gym import spaces
from env.arm_env import ArmEnv


class GymArmEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # ===============================
    # Curriculum configuration
    # ===============================
    STAGE_XY_ONLY = 1
    STAGE_ADD_Z   = 2
    STAGE_FULL    = 3

    def __init__(self, gui=False, stage=STAGE_XY_ONLY):
        super().__init__()

        self.env = ArmEnv(gui=gui)

        # -------- Curriculum stage --------
        self.stage = stage

        # -------- Action space --------
        # Always keep (3,) for PPO compatibility
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # -------- Observation --------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # -------- Episode --------
        self.max_steps = 120
        self.step_count = 0

        # -------- State --------
        self.joint_positions = np.zeros(7)
        self.ee_target = None

        self.prev_distance = None
        self.prev_ee_pos = None

        # -------- Commitment --------
        self.commitment_active = False

    # =====================================================
    # Gym API
    # =====================================================

    def reset(self):
        obs = self.env.reset()

        self.joint_positions = obs[:7].copy()
        self.ee_target = obs[7:10].copy()

        self.prev_distance = self.env.compute_distance()
        self.prev_ee_pos = obs[7:10].copy()

        self.commitment_active = False
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
            "success": success,
            "stage": self.stage
        }

        return obs.astype(np.float32), reward, done, info

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    # =====================================================
    # Action handling
    # =====================================================

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)

        # -------- STAGE 1: XY ONLY --------
        if self.stage == self.STAGE_XY_ONLY:
            ee_delta = np.array([action[0], action[1], 0.0]) * 0.015

        # -------- STAGE 2+: Z allowed near cube --------
        else:
            ee_delta = action * 0.015

            # Z only allowed when close
            if self.prev_distance > 0.25:
                ee_delta[2] = 0.0

        self.ee_target += ee_delta

        # -------- Workspace limits --------
        self.ee_target[0] = np.clip(self.ee_target[0], 0.3, 0.8)
        self.ee_target[1] = np.clip(self.ee_target[1], -0.4, 0.4)
        self.ee_target[2] = np.clip(self.ee_target[2], 0.05, 0.6)

    # =====================================================
    # Simulation
    # =====================================================

    def _simulate(self):
        obs = self.env.step_ik(self.ee_target, self.joint_positions)
        self.joint_positions = obs[:7].copy()
        return obs

    # =====================================================
    # Reward
    # =====================================================

    def _compute_reward(self, obs):
        dist = self.env.compute_distance()

        # -------- Distance reward --------
        distance_reward = 6.0 * (self.prev_distance - dist)

        # -------- Direction reward --------
        ee_pos = obs[7:10]
        obj_pos = obs[10:13]

        ee_vel = ee_pos - self.prev_ee_pos
        self.prev_ee_pos = ee_pos.copy()

        direction = obj_pos - ee_pos
        direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
        directional_reward = np.dot(ee_vel, direction_unit)

        reward = distance_reward + 0.3 * directional_reward

        # -------- STAGE 2: Z encouragement --------
        if self.stage >= self.STAGE_ADD_Z and dist < 0.25:
            reward += 2.5 * max(0.0, self.prev_ee_pos[2] - ee_pos[2])

        # -------- STAGE 3: Commitment --------
        if self.stage == self.STAGE_FULL:
            if dist < 0.12:
                self.commitment_active = True

            if self.commitment_active:
                if dist > self.prev_distance + 0.01:
                    reward -= 1.0  # punish backing off

            # Smoothness penalty (ONLY HERE)
            if ee_vel[2] > 0:   # only penalize upward / sideways jitter
                reward -= 0.02 * np.linalg.norm(ee_vel)
            if dist < 0.15:
                reward += 3.0 * max(0.0, self.prev_ee_pos[2] - ee_pos[2])

        # -------- Success bonuses --------
        if dist < 0.15:
            reward += 1.0
        if dist < 0.10:
            reward += 2.0
        if dist < 0.07:
            reward += 3.0

        self.prev_distance = dist
        return reward + 0.01

    # =====================================================
    # Termination
    # =====================================================

    def _check_done(self):
        success = False
        done = False

        if self.prev_distance < 0.05:
            success = True
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return done, success
