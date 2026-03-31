import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

test_dir = "data/processed/test"
model_path = "models/skin_lesion_model.keras"

model = tf.keras.models.load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_gen.classes
y_probs = model.predict(test_gen).ravel()
y_pred = (y_probs >= 0.5).astype(int)

auc = roc_auc_score(y_true, y_probs)
print("ROC-AUC:", auc)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

with open("results/metrics.txt", "w") as f:
    f.write(f"ROC-AUC: {auc}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred)) + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred, digits=4))