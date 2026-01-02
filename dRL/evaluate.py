from stable_baselines3 import PPO
from env.gym_arm_env import GymArmEnv
import numpy as np

env = GymArmEnv(gui=True)
model = PPO.load("results/drl/ppo_reach")

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    print("Action:", np.round(action, 3))
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
