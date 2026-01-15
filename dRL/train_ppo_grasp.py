from stable_baselines3 import PPO
from env.gym_arm_grasp_env import GymArmGraspEnv

env = GymArmGraspEnv(gui=True)

model = PPO(
    "MlpPolicy",
    env,
    device="cpu",
    verbose=1,
    n_steps=1024,
    batch_size=256,
    learning_rate=3e-4
)

model.learn(total_timesteps=2000)

model.save("ppo_grasp")

print("Grasp training finished")