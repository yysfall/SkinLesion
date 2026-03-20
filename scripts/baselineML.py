import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = "data/processed/train"
val_dir = "data/processed/val"


datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir, target_size=(128,128), batch_size=32, class_mode='binary', shuffle=True)
val_gen = datagen.flow_from_directory(val_dir, target_size=(128,128), batch_size=32, class_mode='binary', shuffle=False)


X_train, y_train = [], []
for i in range(len(train_gen)):
    imgs, labels = train_gen[i]
    X_train.append(imgs.reshape(imgs.shape[0], -1))
    y_train.append(labels)
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

X_val, y_val = [], []
for i in range(len(val_gen)):
    imgs, labels = val_gen[i]
    X_val.append(imgs.reshape(imgs.shape[0], -1))
    y_val.append(labels)
X_val = np.vstack(X_val)
y_val = np.hstack(y_val)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_val)[:,1]


auc = roc_auc_score(y_val, y_pred)
print("Baseline RF ROC-AUC:", auc)