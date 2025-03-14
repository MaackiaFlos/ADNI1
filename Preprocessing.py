import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_image(image_path, ID):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(image, cmap="gray")
    plt.title(f"Original Image {ID}")
    plt.axis("off")
    plt.show()

    image = cv2.resize(image, (128, 128))
    plt.imshow(image, cmap="gray")
    plt.title(f"Resized Image {ID}")
    plt.axis("off")
    plt.show()

    # Normalize the pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0
    plt.imshow(image, cmap="gray")
    plt.title(f"Normalized Image {ID}")
    plt.axis("off")
    plt.show()

    # Apply Gaussian blur for noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
    plt.imshow(image, cmap="gray")
    plt.title(f"Blurred Image {ID}")
    plt.axis("off")
    plt.show()

    # Calculate adaptive Canny thresholds based on the image's median intensity
    median_intensity = np.median(image)
    lower_threshold = int(max(0, 0.7 * median_intensity * 255))
    upper_threshold = int(min(255, 1.3 * median_intensity * 255))

    # Apply Canny edge detection with adaptive thresholds
    edges = cv2.Canny((image * 255).astype(np.uint8), lower_threshold, upper_threshold)
    plt.imshow(edges, cmap="gray")
    plt.title(f"Edge-Detected Image {ID}")
    plt.axis("off")
    plt.show()

    # Normalize the edge-detected image to [0, 1]
    edges = edges.astype(np.float32) / 255.0
    plt.imshow(edges, cmap="gray")
    plt.title(f"Normalized Edge Image {ID}")
    plt.axis("off")
    plt.show()

    # Combine the original image and edges (optional)
    combined = cv2.addWeighted(image, 0.7, edges, 0.3, 0)
    plt.imshow(combined, cmap="gray")
    plt.title(f"Combined Image {ID}")
    plt.axis("off")
    plt.show()

    return edges

def main():
    df = pd.read_csv(r"ADNI1_Complete_1Yr_1.5T.csv")
    for _, row in df.iterrows():
        image_path = f"Middle Slice/{row['Image Data ID']}.png"
        image = preprocess_image(image_path, row['Image Data ID'])

if __name__ == "__main__":
    main()