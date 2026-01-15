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

        p.setPhysicsEngineParameter(
            numSolverIterations=200,
            numSubSteps=5,
            physicsClientId=self.cid
        )

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
        p.changeDynamics(
            self.object, -1,
            mass=0.3,
            lateralFriction=5.0,
            rollingFriction=1.0,
            spinningFriction=1.0,
            restitution=0.0,
            linearDamping=0.05,
            angularDamping=0.05,
            contactStiffness=200,
            contactDamping=100,
            physicsClientId=self.cid
        )

        num = p.getNumJoints(self.robot, physicsClientId=self.cid)

        for i in range(num):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            print(i, info[1].decode("utf-8"))


        # Franka Panda
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 10]
        self.ee_link = 11

        for j in self.arm_joints:
            p.changeDynamics(self.robot, j, linearDamping=0.04, angularDamping=0.04, physicsClientId=self.cid)
        
        for j in self.gripper_joints:
            p.changeDynamics(
                self.robot, j,
                lateralFriction=2.0,
                restitution=0.0,
                linearDamping=0.04,
                angularDamping=0.04,
                contactStiffness=150,
                contactDamping=40,
                physicsClientId=self.cid
            )

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
                force=60,
                physicsClientId=self.cid
            )

        p.stepSimulation(physicsClientId=self.cid)
        return self.get_observation()

    def step_ik(self, target_ee_pos, current_joint_pos):
        obj_pos = p.getBasePositionAndOrientation(self.object, physicsClientId=self.cid)[0]
        joint_targets = self.ik.solve(target_ee_pos, current_joint_pos, obj_pos)

        for _ in range(8):
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

    def control_gripper(self, cmd):
        if cmd > 0:
            self.close_gripper()
        else:
            self.open_gripper()

    def close_gripper(self):
        for j in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                maxVelocity=0.01,
                force=100,
                physicsClientId=self.cid
            )

    def open_gripper(self):
        for j in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                maxVelocity=0.02,
                force=30,
                physicsClientId=self.cid
            )

    def is_grasping(self):
        contacts = p.getContactPoints(self.robot, self.object, physicsClientId=self.cid)

        finger_contacts = [
            c for c in contacts
            if c[3] in self.gripper_joints or c[4] in self.gripper_joints
        ]

        obj_z = self.get_object_height()

        if len(finger_contacts) >= 2 and obj_z > 0.05:
            return True

        return False


    def get_object_height(self):
        return p.getBasePositionAndOrientation(self.object, physicsClientId=self.cid)[0][2]