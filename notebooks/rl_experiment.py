from src.rl.environment import ThresholdEnv
from src.rl.agent import QAgent
import numpy as np
import matplotlib.pyplot as plt

probs = np.random.rand(100)
labels = np.random.randint(0, 2, 100)

env = ThresholdEnv(probs, labels)
agent = QAgent(n_actions=5)

rewards = []

for episode in range(50):
    action = agent.select_action()
    reward = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)

plt.plot(rewards)
plt.title("RL Learning Curve")
plt.show()