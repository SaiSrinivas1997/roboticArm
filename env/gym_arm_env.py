import gym
import numpy as np
from gym import spaces
from env.arm_env import ArmEnv
import time

class GymArmEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False):
        super().__init__()

        self.env = ArmEnv(gui=gui)

        # Action = delta end-effector position
        self.max_vel = 0.5
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

        self.dt = 1.0 / 240.0 
        self.max_steps = 100
        self.step_count = 0

        self.joint_positions = np.zeros(7)
        self.prev_distance = None

        self.ee_target = None

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

        # -----------------------
        # Action = EE delta
        # -----------------------
        action = np.clip(action, -1.0, 1.0)
        ee_delta = action * 0.05   # meters per step

        # 🔥 KEY FIX: accumulate target
        self.ee_target += ee_delta

        # Optional Z clamp (prevents table crash)
        self.ee_target[2] = np.clip(self.ee_target[2], 0.05, 0.8)

        # Apply IK (THIS IS THE KEY LINE)
        obs = self.env.step_ik(self.ee_target)

        # Update stored state
        self.joint_positions = obs[:7].copy()

        # -----------------------
        # Reward
        # -----------------------
        dist = self.env.compute_distance()
        distance_reward = 5.0 * (self.prev_distance - dist)

        ee_vel = obs[7:10] - self.prev_ee_pos
        self.prev_ee_pos = obs[7:10].copy()

        direction = obs[10:13] - obs[7:10]
        direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
        directional_reward = np.dot(ee_vel, direction_unit)

        smoothness_penalty = 0.03 * np.linalg.norm(action)

        reward = distance_reward + 0.3 * directional_reward - smoothness_penalty + 0.01
        self.prev_distance = dist

        # -----------------------
        # Termination
        # -----------------------
        done = False
        success = False

        if dist < 0.05:
            reward += 10.0
            done = True
            success = True

        if self.step_count >= self.max_steps:
            done = True

        info = {"distance": dist, "success": success}

        time.sleep(0.03)

        return obs.astype(np.float32), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

