import time
import numpy as np
from env.arm_env import ArmEnv

env = ArmEnv(gui=True)
obs = env.reset()

for t in range(500):
    action = 0.3 * np.sin(0.02 * t) * np.ones(7)
    obs = env.step(action)

    if t % 20 == 0:
        print("Distance:", env.compute_distance())

    time.sleep(0.01)
