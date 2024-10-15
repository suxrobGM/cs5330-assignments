# Author: Sukhrob Ilyosbekov
# Date: 10/14/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core import get_image_path

def main() -> None:
    """
    Convert a regular image to a night vision image.
    """

    # Get the image path from the dataset/images directory and load the image
    img_path = get_image_path("night_street.jpg")
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    # This simulates the smoothing effect of night vision devices
    blurred = cv2.GaussianBlur(grayed, (5, 5), 0)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This improves the visibility of important elements in low-light conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
 
    # Apply edge detection using the Canny algorithm
    edges = cv2.Canny(blurred, 50, 150)

    # Invert the edges image
    # So that edges are white on a black background
    edges = cv2.bitwise_not(edges)

    # Blend the enhanced image and the inverted edges to highlight important features
    # This ensures that edges (important elements) are bright in the final image
    blended = cv2.bitwise_and(enhanced, edges)

    # Apply a color map to simulate the green hue of night vision devices
    # COLORMAP_BONE, COLORMAP_OCEAN, COLORMAP_SUMMER can give greenish tones
    colored = cv2.applyColorMap(blended, cv2.COLORMAP_OCEAN)

    # Adjust brightness and contrast
    # Alpha controls contrast (1.0-3.0), beta controls brightness (0-100)
    adjusted = cv2.convertScaleAbs(colored, alpha=1.5, beta=10)

    display_images(img, edges, adjusted)


def display_images(org_img: np.ndarray, edges: np.ndarray, night: np.ndarray) -> None:
    """
    Display the original, edges, and night vision images in a 1x3 grid.
    Args:
        org_img (np.ndarray): The original image.
        edges (np.ndarray): The edges image.
        night (np.ndarray): The filtered image.
    """

    # Display the images in a 1x3 grid
    _, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes: list[Axes] # Type hint for axes

    axes[0].imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGBA))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Edges Image")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(night, cv2.COLOR_BGR2RGBA))
    axes[2].set_title("Night Vision Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
