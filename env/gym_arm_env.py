import gym
import numpy as np
from gym import spaces

from env.arm_env import ArmEnv


class GymArmEnv(gym.Env):
    """
    Gym wrapper for ArmEnv
    Task: Reach the cube (no grasping)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False):
        super().__init__()

        self.env = ArmEnv(gui=gui)

        # -----------------------
        # Action space
        # -----------------------
        # Action = small joint position deltas
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # -----------------------
        # Observation space
        # [joint_pos(7), ee_pos(3), obj_pos(3)]
        # -----------------------
        self.obs_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # -----------------------
        # Control parameters
        # -----------------------
        self.delta_scale = 0.02     # radians per step (CRITICAL)
        self.max_steps = 200
        self.step_count = 0

        # Franka Panda joint limits
        self.joint_min = np.array(
            [-2.9, -1.8, -2.9, -3.1, -2.9, -0.1, -2.9]
        )
        self.joint_max = np.array(
            [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9]
        )

        self.joint_positions = np.zeros(7)

    # -----------------------
    # Reset
    # -----------------------
    def reset(self):
        obs = self.env.reset()

        self.joint_positions = obs[:7].copy()
        self.prev_distance = self.env.compute_distance()
        self.step_count = 0

        return obs.astype(np.float32)

    # -----------------------
    # Step
    # -----------------------
    def step(self, action):
        self.step_count += 1

        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Convert action → joint delta
        delta = action * self.delta_scale
        self.joint_positions = self.joint_positions + delta

        # Enforce joint limits
        self.joint_positions = np.clip(
            self.joint_positions,
            self.joint_min,
            self.joint_max
        )

        # Step simulation (multiple for stability)
        for _ in range(5):
            obs = self.env.step(self.joint_positions)

        # -----------------------
        # Reward
        # -----------------------
        dist = self.env.compute_distance()

        # Distance improvement reward
        reward = self.prev_distance - dist

        # Penalize moving away
        if dist > self.prev_distance:
            reward -= 0.05

        # Small living reward
        reward += 0.01

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

        info = {
            "distance": dist,
            "success": success
        }

        return obs.astype(np.float32), reward, done, info

    # -----------------------
    def render(self, mode="human"):
        pass

    def close(self):
        self.env = None

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
