import pybullet as p

class IKSolver:
    def __init__(self, robot_id, ee_link, arm_joints):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.arm_joints = arm_joints

        # Panda joint limits (safe)
        self.lower_limits = [-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9]
        self.upper_limits = [ 2.9,  1.8,  2.9,  0.0,  2.9,  3.7,  2.9]
        self.joint_ranges = [u - l for u, l in zip(self.upper_limits, self.lower_limits)]
        self.rest_pose = [0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8]


    def solve(self, target_ee_pos, current_joint_pos):
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            target_ee_pos,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_pose,
            maxNumIterations=50,
            residualThreshold=1e-4
        )

        # Return only arm joints (ignore gripper)
        return joint_targets[:len(self.arm_joints)]
