import numpy as np

class ThresholdEnv:
    def __init__(self, probs, labels):
        self.probs = probs
        self.labels = labels
        self.thresholds = np.linspace(0.3, 0.7, 5)

    def step(self, action):
        threshold = self.thresholds[action]
        preds = (self.probs >= threshold).astype(int)

        tp = ((preds == 1) & (self.labels == 1)).sum()
        fp = ((preds == 1) & (self.labels == 0)).sum()
        fn = ((preds == 0) & (self.labels == 1)).sum()

        sensitivity = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + (self.labels == 0).sum() + 1e-6)

        reward = sensitivity - 0.5 * fpr
        return reward