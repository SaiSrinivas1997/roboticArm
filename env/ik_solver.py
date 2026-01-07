import pybullet as p

class IKSolver:
    def __init__(self, robot_id, ee_link, arm_joints):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.arm_joints = arm_joints

    def solve(self, target_ee_pos):
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            target_ee_pos
        )

        # Return only arm joints (ignore gripper)
        return joint_targets[:len(self.arm_joints)]
