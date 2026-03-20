import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def extract_features(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    return img.flatten() / 255.0

image_paths = ["img1.jpg", "img2.jpg"]
labels = [0, 1]

X = np.array([extract_features(p) for p in image_paths])
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_val)[:,1]
roc = roc_auc_score(y_val, y_probs)

print("Baseline ML ROC-AUC:", roc)