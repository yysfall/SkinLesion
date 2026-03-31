import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        if isinstance(preds, (list, tuple)):
            if len(preds) == 1:
                preds = preds[0]
            else:
                preds = tf.stack(preds, axis=-1)

        preds = tf.convert_to_tensor(preds)

        if preds.shape.ndims == 1:
            loss = preds
        else:
            loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize((img_arr.shape[1], img_arr.shape[0]), Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_arr / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)

    superimposed = np.uint8((1 - alpha) * img_arr + alpha * heatmap_colored)
    return img_arr, superimposed


def save_gradcam_result(original, superimposed, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()