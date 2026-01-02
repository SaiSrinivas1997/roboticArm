import gym
import numpy as np
from gym import spaces

from env.arm_env import ArmEnv


class GymArmEnv(gym.Env):
    """
    Gym wrapper for ArmEnv
    Task: Reach the object (no grasping yet)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False):
        super().__init__()

        self.env = ArmEnv(gui=gui)

        # ---- Action space ----
        # 7 joint velocities
        self.max_vel = 0.5  # rad/s
        self.action_space = spaces.Box(
            low=-self.max_vel,
            high=self.max_vel,
            shape=(7,),
            dtype=np.float32
        )

        # ---- Observation space ----
        # [joint_pos(7), ee_pos(3), obj_pos(3)]
        self.obs_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # ---- Control params ----
        self.dt = 1.0 / 240.0
        self.max_steps = 200
        self.step_count = 0

        self.joint_positions = np.zeros(7)

    def reset(self):
        obs = self.env.reset()

        self.joint_positions[:] = obs[:7]
        self.step_count = 0

        return obs.astype(np.float32)

    def step(self, action):
        self.step_count += 1

        # Clip action
        action = np.clip(action, -self.max_vel, self.max_vel)

        # Integrate joint positions
        self.joint_positions += action * self.dt

        # Step simulation
        obs = self.env.step(self.joint_positions)

        # Compute reward
        dist = self.env.compute_distance()
        reward = -dist

        # Termination conditions
        done = False
        success = False

        if dist < 0.05:
            done = True
            success = True
            reward += 10.0  # success bonus

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "distance": dist,
            "success": success
        }

        return obs.astype(np.float32), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.env = None
