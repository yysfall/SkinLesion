import numpy as np
import matplotlib.pyplot as plt

# Replace these with actual CNN probabilities later if available
np.random.seed(42)
y_probs = np.random.rand(100)
y_true = np.random.randint(0, 2, 100)

thresholds = np.linspace(0.0, 1.0, 51)
rewards = []

best_threshold = 0.5
best_reward = -1e9

for thr in thresholds:
    y_pred = (y_probs >= thr).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    reward = (2 * tp) + (1 * tn) - (1 * fp) - (3 * fn)
    rewards.append(reward)

    if reward > best_reward:
        best_reward = reward
        best_threshold = thr

print(f"Best threshold: {best_threshold}")
print(f"Best reward: {best_reward}")

plt.figure()
plt.plot(thresholds, rewards, marker="o")
plt.xlabel("Threshold")
plt.ylabel("Reward")
plt.title("Threshold Tuning Reward Curve")
plt.savefig("results/plots/rl_reward_curve.png")
plt.close()