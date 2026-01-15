import time
import numpy as np
from stable_baselines3 import PPO
from env.gym_arm_grasp_env import GymArmGraspEnv

MODEL_PATH = "ppo_grasp.zip"   # change if different

def evaluate(n_episodes=20, gui=True):

    env = GymArmGraspEnv(gui=gui)

    model = PPO.load(MODEL_PATH, device="cpu")

    success_count = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0

        print(f"\n=== Episode {ep+1} ===")

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            step += 1

            if gui:
                time.sleep(0.01)

        if info.get("success", False):
            success_count += 1
            print("✅ SUCCESS")
        else:
            print("❌ FAIL")

        print(f"Steps: {step}")
        print(f"Grasped: {info.get('grasped', False)}")

    print("\n==============================")
    print(f"Success rate: {success_count}/{n_episodes}")
    print("==============================")

    env.close()


if __name__ == "__main__":
    evaluate(n_episodes=10, gui=True)
