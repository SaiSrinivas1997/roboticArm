from stable_baselines3 import PPO
from env.gym_arm_env import GymArmEnv
import numpy as np
import time

env = GymArmEnv(gui=True, stage=GymArmEnv.STAGE_FULL)

model = PPO.load(
    "results/drl/ppo_reach",
    env=env,
    device="cpu"   # 👈 ADD THIS
)


obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    print("Action:", np.round(action, 3))

    obs, reward, done, info = env.step(action)
    time.sleep(0.06)

    if done:
        print("Episode finished")
        print("Final distance to cube:", round(info["distance"], 4))
        print("Success:", info["success"])
        print("-" * 40)

        # obs = env.reset()
