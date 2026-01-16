import gym
import numpy as np
import pybullet as p
from gym import spaces
from env.arm_env import ArmEnv

class GymArmGraspEnv(gym.Env):

    def __init__(self, gui=False):
        super().__init__()

        self.env = ArmEnv(gui=gui)

        # Action: [x_cmd, z_cmd, gripper_cmd]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([ 1.0,  1.0,  1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.max_steps = 80
        self.step_count = 0

        self.ee_target = None
        self.grasp_phase = "approach"
        self.hold_steps = 0

    # -------------------------------------------------

    def reset(self):
        obs = self.env.reset()

        obj_pos = obs[10:13]

        # Start above cube
        self.ee_target = obj_pos + np.array([0.0, 0.0, 0.08])

        self.joint_positions = obs[:7].copy()
        self.step_count = 0

        self.hold_steps = 0
        self.grasp_phase = "approach"

        return obs.astype(np.float32)

    # -------------------------------------------------

    def step(self, action):
        self.step_count += 1

        x_cmd = action[0]
        z_cmd = action[1]
        grip_cmd = action[2]

        obs = self.env.get_observation()
        obj_pos = obs[10:13]
        ee_pos  = obs[7:10]

        # -------------------------
        # HOLD LOGIC
        # -------------------------
        if self.env.is_grasping():
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        # -------------------------
        # PHASE LOGIC
        # -------------------------
        if self.hold_steps > 12:
            phase = "lift"
        elif self.hold_steps > 4:
            phase = "settle"
        else:
            phase = "approach"

        # -------------------------
        # MOTION CONTROL
        # -------------------------
        xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])

        if phase == "approach":
            self.ee_target[0] += 0.002 * x_cmd

            dy = obj_pos[1] - ee_pos[1]
            dy = np.clip(dy, -0.002, 0.002)
            self.ee_target[1] = ee_pos[1] + dy

            if xy_dist < 0.01:
                self.ee_target[2] -= 0.0015

        elif phase == "settle":
            # freeze motion to stabilize grasp
            self.ee_target[:] = ee_pos

        elif phase == "lift":
            self.ee_target[2] += 0.0008

        self.ee_target[2] = np.clip(self.ee_target[2], 0.02, 0.25)

        # -------------------------
        # GRIPPER (sticky)
        # -------------------------
        centered = np.linalg.norm(ee_pos[:2] - obj_pos[:2]) < 0.006
        height_ok = abs(ee_pos[2] - obj_pos[2]) < 0.015

        if centered and height_ok:
            self.env.close_gripper()
        else:
            self.env.open_gripper()

        if self.env.is_grasping():
            self.ee_target[0] = ee_pos[0]
            self.ee_target[1] = ee_pos[1]

        # -------------------------
        # IK TARGET
        # -------------------------
        target = self.ee_target.copy()
        if not self.env.is_grasping():
            target[2] = obj_pos[2] + 0.06

        obs = self.env.step_ik(target, self.joint_positions)
        self.joint_positions = obs[:7].copy()

        if not self.env.is_grasping():
            p.applyExternalForce(self.env.object, -1, [0,0,-0.5], [0,0,0], p.WORLD_FRAME)


        reward = self._compute_reward(obs)
        done, success = self._check_done()

        info = {
            "grasped": self.env.is_grasping(),
            "success": success,
            "phase": phase
        }

        if self.step_count % 20 == 0:
            print(f"[DBG] phase={phase}, ee={obs[7:10].round(3)}, grasped={info['grasped']}")

        return obs.astype(np.float32), reward, done, info


    # -------------------------------------------------

    def _compute_reward(self, obs):
        ee_pos = obs[7:10]
        obj_pos = obs[10:13]

        dist = np.linalg.norm(ee_pos - obj_pos)

        reward = 1.0 - dist * 3.0

        # y alignment reward
        y_err = abs(ee_pos[1] - obj_pos[1])
        reward += 3.0 * np.exp(- (y_err / 0.01)**2)

        # top-down preference
        if ee_pos[2] > obj_pos[2]:
            reward += 1.5

        # proximity bonuses
        if dist < 0.05:
            reward += 3.0
        if dist < 0.03:
            reward += 3.0

        # grasp reward
        if self.env.is_grasping():
            reward += 10.0

        # hold reward
        if self.hold_steps > 0:
            reward += 0.5 * self.hold_steps

        return reward

    # -------------------------------------------------

    def _check_done(self):
        done = False
        success = False

        if self.hold_steps >= 20:
            success = True
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return done, success
