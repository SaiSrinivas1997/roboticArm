import pybullet as p
import time
import numpy as np
from env.arm_env import ArmEnv

# Start environment in GUI mode
env = ArmEnv(gui=True)
env.reset()
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.4, 0, 0.4]
)


print("""
Teleoperation Controls:
Joint 0: 1 / Q
Joint 1: 2 / W
Joint 2: 3 / E
Joint 3: 4 / R
Joint 4: 5 / T
Joint 5: 6 / Y
Joint 6: 7 / U
SPACE: Quit
""")

joint_positions = np.zeros(7)
dt = 1.0 / 240.0
joint_vel = np.zeros(7)
speed = 0.6   # rad/s

while True:
    keys = p.getKeyboardEvents()
    joint_vel[:] = 0.0  # reset every frame

    # Exit
    if p.B3G_SPACE in keys:
        print("Exiting teleop")
        break

    # Joint 0
    if ord('1') in keys: joint_vel[0] = +speed
    if ord('q') in keys: joint_vel[0] = -speed

    # Joint 1
    if ord('2') in keys: joint_vel[1] = +speed
    if ord('w') in keys: joint_vel[1] = -speed

    # Joint 2
    if ord('3') in keys: joint_vel[2] = +speed
    if ord('e') in keys: joint_vel[2] = -speed

    # Joint 3
    if ord('4') in keys: joint_vel[3] = +speed
    if ord('r') in keys or ord('R') in keys: joint_vel[3] = -speed

    # Joint 4
    if ord('5') in keys: joint_vel[4] = +speed
    if ord('t') in keys: joint_vel[4] = -speed

    # Joint 5
    if ord('6') in keys: joint_vel[5] = +speed
    if ord('y') in keys: joint_vel[5] = -speed

    # Joint 6
    if ord('7') in keys: joint_vel[6] = +speed
    if ord('u') in keys: joint_vel[6] = -speed

    if ord('o') in keys:
        env.open_gripper()

    if ord('c') in keys:
        env.close_gripper()

    # Integrate position manually
    joint_positions += joint_vel * dt

    env.step(joint_positions)
    time.sleep(dt)


