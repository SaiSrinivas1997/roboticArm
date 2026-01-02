import numpy as np
import os
from env.gym_arm_env import GymArmEnv

# =========================
# Config
# =========================
EPISODES = 300
NOISE_SCALE = 0.02
MAX_STEPS = 200
SAVE_PATH = "results/classic_rl"

os.makedirs(SAVE_PATH, exist_ok=True)

# =========================
# Environment
# =========================
env = GymArmEnv(gui=False)

obs_dim = env.observation_space.shape[0]   # 13
act_dim = env.action_space.shape[0]         # 7
max_vel = env.max_vel

# =========================
# Linear Policy: action = W @ obs
# =========================
W = np.random.randn(act_dim, obs_dim) * 0.1

reward_history = []

# =========================
# Run one episode
# =========================
def run_episode(W):
    obs = env.reset()
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        action = W @ obs
        action = np.clip(action, -max_vel, max_vel)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward

# =========================
# Training Loop (Hill Climbing)
# =========================
best_reward = -1e9

for ep in range(EPISODES):
    # Evaluate current policy
    reward = run_episode(W)

    # Try noisy version
    W_try = W + NOISE_SCALE * np.random.randn(*W.shape)
    reward_try = run_episode(W_try)

    # Accept if better
    if reward_try > reward:
        W = W_try
        reward = reward_try

    reward_history.append(reward)
    best_reward = max(best_reward, reward)

    if ep % 20 == 0:
        print(f"[Episode {ep}] Reward: {reward:.3f}")

# =========================
# Save results
# =========================
np.save(f"{SAVE_PATH}/rewards.npy", np.array(reward_history))
np.save(f"{SAVE_PATH}/W.npy", W)

env.close()
print("Training finished.")
