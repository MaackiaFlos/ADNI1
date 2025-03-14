import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = cv2.GaussianBlur(img, (5, 5), 0)

    median_intensity = np.median(img)
    lower_threshold = int(max(0, 0.7 * median_intensity * 255))
    upper_threshold = int(min(255, 1.3 * median_intensity * 255))

    img = cv2.Canny((img * 255).astype(np.uint8), lower_threshold, upper_threshold)
    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    return img

def superimpose_heatmap(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.colormaps.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

def main():
    model = tf.keras.models.load_model("Model UNet/Model_UNet.keras")
    model.summary()
    class_labels = ["Alzheimer's Disease (AD)", "Mild Cognitive Impairment (MCI)", "Cognitively Normal (CN)"]

    last_conv_layer_name = "conv2d_2"

    # Create an output directory to save images
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(r"ADNI1_Complete_1Yr_1.5T.csv")
    for _, row in df.iterrows():
        image_path = f"Middle Slice/{row['Image Data ID']}.png"  # Replace with your image path
        img_array = load_and_preprocess_image(image_path)

        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        superimposed_img = superimpose_heatmap(img, heatmap)

        # Display and save the original image
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        original_image_path = os.path.join(output_dir, f"{row['Image Data ID']}_original.png")
        plt.savefig(original_image_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Display and save the heatmap
        plt.figure(figsize=(5, 5))
        heatmap_display = plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        plt.colorbar(heatmap_display, fraction=0.046, pad=0.04)
        heatmap_image_path = os.path.join(output_dir, f"{row['Image Data ID']}_heatmap.png")
        plt.savefig(heatmap_image_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Display and save the superimposed image
        plt.figure(figsize=(5, 5))
        plt.imshow(superimposed_img)
        plt.title(f"Superimposed Image\nPredicted: {predicted_class}")
        plt.axis('off')
        superimposed_image_path = os.path.join(output_dir, f"{row['Image Data ID']}_superimposed.png")
        plt.savefig(superimposed_image_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved images for {row['Image Data ID']}")

if __name__ == "__main__":
    main()