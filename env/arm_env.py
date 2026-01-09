import pybullet as p
import pybullet_data
import numpy as np
from env.ik_solver import IKSolver


class ArmEnv:
    def __init__(self, gui=False):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / 1000.0, physicsClientId=self.cid)

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=self.cid
        )
        self.object = p.loadURDF(
            "cube_small.urdf",
            [0.6, 0.0, 0.03],
            physicsClientId=self.cid
        )

        # Franka Panda
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 10]
        self.ee_link = 11

        self.ik = IKSolver(
            robot_id=self.robot,
            ee_link=self.ee_link,
            arm_joints=self.arm_joints,
            cid = self.cid
        )

        self.reset()

    def reset(self):
        for j in self.arm_joints:
            p.resetJointState(self.robot, j, 0.0, physicsClientId=self.cid)

        for j in self.gripper_joints:
            p.resetJointState(self.robot, j, 0.04, physicsClientId=self.cid)

        p.stepSimulation(physicsClientId=self.cid)
        return self.get_observation()

    def step_joints(self, joint_targets):
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[i],
                force=200,
                physicsClientId=self.cid
            )

        p.stepSimulation(physicsClientId=self.cid)
        return self.get_observation()

    def step_ik(self, target_ee_pos, current_joint_pos):
        joint_targets = self.ik.solve(target_ee_pos, current_joint_pos)
        for _ in range(5):
            self.step_joints(joint_targets)
        return self.get_observation()


    def get_observation(self):
        joint_states = p.getJointStates(self.robot, self.arm_joints, physicsClientId=self.cid)
        joint_pos = np.array([s[0] for s in joint_states])

        ee_pos = np.array(
            p.getLinkState(self.robot, self.ee_link, physicsClientId=self.cid)[0]
        )

        obj_pos = np.array(
            p.getBasePositionAndOrientation(self.object, physicsClientId=self.cid)[0]
        )

        return np.concatenate([joint_pos, ee_pos, obj_pos])

    def compute_distance(self):
        ee = np.array(p.getLinkState(self.robot, self.ee_link, physicsClientId=self.cid)[0])
        obj = np.array(p.getBasePositionAndOrientation(self.object, physicsClientId=self.cid)[0])
        return np.linalg.norm(ee - obj)

    def close(self):
        p.disconnect(self.cid)
