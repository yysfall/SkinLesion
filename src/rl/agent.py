import numpy as np

class QAgent:
    def __init__(self, n_actions):
        self.q = np.zeros(n_actions)
        self.lr = 0.1

    def select_action(self):
        return np.argmax(self.q)

    def update(self, action, reward):
        self.q[action] += self.lr * (reward - self.q[action])