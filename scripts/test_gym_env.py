import numpy as np
from env.gym_arm_env import GymArmEnv

env = GymArmEnv(gui=True)
obs = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"dist={info['distance']:.3f}, reward={reward:.3f}")
    if done:
        break
