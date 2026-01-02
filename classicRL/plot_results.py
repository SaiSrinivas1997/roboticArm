import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("results/classic_rl/rewards.npy")

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Classical RL – Reaching Task")
plt.grid(True)
plt.show()
