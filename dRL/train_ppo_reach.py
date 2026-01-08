import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.gym_arm_env import GymArmEnv

# =========================
# Config
# =========================
SAVE_DIR = "results/drl"
os.makedirs(SAVE_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 50_000

# =========================
# Vectorized Environment
# =========================
env = make_vec_env(
    lambda: GymArmEnv(gui=False, stage=GymArmEnv.STAGE_FULL),
    n_envs=4
)

# =========================
# PPO Model
# =========================
model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cpu",              # CPU is correct for MLP
    verbose=1,

    # PPO stability (important)
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,

    # Exploration (helps early reaching)
    ent_coef=0.01
)

# =========================
# Train
# =========================
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# =========================
# Save
# =========================
model.save(f"{SAVE_DIR}/ppo_reach")

env.close()
print("PPO reach training finished.")
