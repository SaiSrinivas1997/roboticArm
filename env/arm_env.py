import pybullet as p
import pybullet_data
import numpy as np

class ArmEnv:
    def __init__(self, gui=False):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True
        )

        self.object = p.loadURDF(
            "cube_small.urdf",
            [0.6, 0.0, 0.03]
        )

        # ✅ Franka Panda specifics
        self.arm_joints = list(range(7))      # panda_joint1 … panda_joint7
        self.gripper_joints = [9, 10]          # fingers
        self.ee_link = 11                      # panda_hand

        self.reset()

    def reset(self):
        # Reset arm
        for j in self.arm_joints:
            p.resetJointState(self.robot, j, 0.0)

        # Open gripper by default
        for j in self.gripper_joints:
            p.resetJointState(self.robot, j, 0.04)

        p.stepSimulation()
        return self.get_observation()

    def step(self, joint_targets):
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[i],
                force=200
            )

        p.stepSimulation()
        return self.get_observation()

    def get_observation(self):
        joint_states = p.getJointStates(self.robot, self.arm_joints)
        joint_pos = np.array([s[0] for s in joint_states])

        ee_pos = np.array(
            p.getLinkState(self.robot, self.ee_link)[0]
        )

        obj_pos = np.array(
            p.getBasePositionAndOrientation(self.object)[0]
        )

        return np.concatenate([joint_pos, ee_pos, obj_pos])

    def compute_distance(self):
        ee = np.array(p.getLinkState(self.robot, self.ee_link)[0])
        obj = np.array(p.getBasePositionAndOrientation(self.object)[0])
        return np.linalg.norm(ee - obj)

    def check_contact(self):
        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=self.object
        )
        return len(contacts) > 0

    def open_gripper(self):
        for j in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=50
            )

    def close_gripper(self):
        for j in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=50
            )
