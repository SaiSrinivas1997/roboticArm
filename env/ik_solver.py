import pybullet as p
import numpy as np

class IKSolver:
    def __init__(self, robot_id, ee_link, arm_joints, cid):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.arm_joints = arm_joints
        self.cid = cid

        # Panda joint limits (safe)
        self.lower_limits = [-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9]
        self.upper_limits = [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9]
        self.joint_ranges = [u - l for u, l in zip(self.upper_limits, self.lower_limits)]
        self.rest_pose = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]


    def solve(self, target_ee_pos, current_joint_pos, obj_pos=None):

        # Top-down stable orientation
        desired_orn = p.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        rest = list(current_joint_pos)

        if obj_pos is not None:
            xy_dist = np.linalg.norm(target_ee_pos[:2] - obj_pos[:2])

            # Smooth elbow bias
            if xy_dist < 0.10:
                rest[3] = -2.2   # strong bend near object
            else:
                rest[3] = -1.2   # mild bend far away

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            target_ee_pos,
            desired_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=rest,
            maxNumIterations=150,
            residualThreshold=1e-5,
            physicsClientId=self.cid
        )

        return joint_targets[:len(self.arm_joints)]










