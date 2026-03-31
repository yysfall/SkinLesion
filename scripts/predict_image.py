import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from scripts.gradcam import (
    get_last_conv_layer_name,
    make_gradcam_heatmap,
    overlay_gradcam,
    save_gradcam_result
)

IMG_SIZE = (224, 224)
MODEL_PATH = "models/skin_lesion_model.keras"
OUTPUT_DIR = "results/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img_array = load_and_preprocess(img_path)

    probs = model.predict(img_array, verbose=0)

    if isinstance(probs, (np.ndarray,)):
        if probs.ndim == 2 and probs.shape[1] == 1:
            prob = float(probs[0, 0])
        elif probs.ndim == 2 and probs.shape[1] == 2:
            # if model is binary softmax
            prob = float(probs[0, 1])
        elif probs.ndim == 1:
            prob = float(probs[0])
        else:
            raise ValueError(f"Unsupported prediction shape: {probs.shape}")
    else:
        prob = float(probs[0])

    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = prob if prob >= 0.5 else 1 - prob

    last_conv_layer = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    original, gradcam_img = overlay_gradcam(img_path, heatmap)

    output_path = os.path.join(OUTPUT_DIR, "gradcam_result.png")
    save_gradcam_result(original, gradcam_img, output_path)

    return label, confidence, output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_image.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    label, confidence, output_path = predict_image(img_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Grad-CAM saved to: {output_path}")