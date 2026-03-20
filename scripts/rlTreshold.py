import numpy as np


y_probs = np.random.rand(100)
y_true = np.random.randint(0,2,100)
thresholds = np.linspace(0,1,50)
best_thr, best_reward = 0, -np.inf

for thr in thresholds:
    y_pred = (y_probs > thr).astype(int)
    reward = np.sum((y_pred==y_true)*np.where(y_true==1, 1, 0.5))  # asymmetric reward
    if reward > best_reward:
        best_reward = reward
        best_thr = thr

print("Best threshold:", best_thr, "Reward:", best_reward)