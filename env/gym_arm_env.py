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
        # self.stage = stage
        self.stage = GymArmEnv.STAGE_ADD_Z

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

        if self.prev_distance < 0.4:
            action[0] *= 0.3
            action[1] *= 0.3

        # -------- STAGE 1: XY ONLY --------
        if self.stage == self.STAGE_XY_ONLY:
            ee_delta = np.array([action[0], action[1], 0.0]) * 0.015

        # -------- STAGE 2+: Z allowed near cube --------
        else:
            ee_delta = action * 0.01
            ee_delta[2] = np.clip(ee_delta[2], -0.006, 0.004)



        self.ee_target += ee_delta

        # -------- Workspace limits --------
        self.ee_target[0] = np.clip(self.ee_target[0], 0.3, 0.7)
        self.ee_target[1] = np.clip(self.ee_target[1], -0.3, 0.3)
        self.ee_target[2] = np.clip(self.ee_target[2], 0.02, 0.8)

    # =====================================================
    # Simulation
    # =====================================================

    def _simulate(self):
        obs = self.env.step_ik(self.ee_target, self.joint_positions)
        obs[7]  = np.clip(obs[7],  0.3, 0.75)
        obs[8]  = np.clip(obs[8], -0.35, 0.35)
        self.joint_positions = obs[:7].copy()
        return obs

    # =====================================================
    # Reward
    # =====================================================

    def _compute_reward(self, obs):
        dist = self.env.compute_distance()

        ee_pos = obs[7:10]
        obj_pos = obs[10:13]

        ee_vel = ee_pos - self.prev_ee_pos

        direction = obj_pos - ee_pos
        direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
        directional_reward = np.dot(ee_vel, direction_unit)

        distance_reward = 6.0 * (self.prev_distance - dist)

        reward = distance_reward + 0.3 * directional_reward

        # Z descent encouragement
        if dist < 0.3:
            reward += 8.0 * max(0.0, self.prev_ee_pos[2] - ee_pos[2])

        # Final touch encouragement
        if dist < 0.12:
            reward += 6.0 * max(0.0, obj_pos[2] + 0.02 - ee_pos[2])

        # Commitment
        if self.stage == self.STAGE_FULL:
            if dist < 0.12:
                self.commitment_active = True

            if self.commitment_active and dist > self.prev_distance + 0.01:
                reward -= 1.0

            if dist < 0.15:
                reward += 3.0 * max(0.0, self.prev_ee_pos[2] - ee_pos[2])

        # Success bonuses
        if dist < 0.15:
            reward += 1.0
        if dist < 0.10:
            reward += 2.0
        if dist < 0.07:
            reward += 3.0

        if dist < 0.05 and abs(ee_pos[2] - obj_pos[2]) < 0.03:
            reward += 50.0

        self.prev_ee_pos = ee_pos.copy()
        self.prev_distance = dist

        if self.step_count % 20 == 0:
            print(
                f"[DEBUG] dist={dist:.3f}, ee={ee_pos.round(3)}, obj={obj_pos.round(3)}, vel={ee_vel.round(3)}"
            )

        return reward + 0.01


    # =====================================================
    # Termination
    # =====================================================

    def _check_done(self):
        success = False
        done = False

        ee_z = self.prev_ee_pos[2]
        obj_z = self.env.get_observation()[12]

        if self.prev_distance < 0.05 and abs(ee_z - obj_z) < 0.03:
            success = True
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return done, success
