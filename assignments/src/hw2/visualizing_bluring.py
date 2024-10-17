# Author: Sukhrob Ilyosbekov
# Date: 10/15/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core import get_image_path

def run() -> None:
    """
    Apply Gaussian blur to an image and visualize the effect of different kernel sizes.
    """

    # Get the image path from the dataset/images directory and load the image
    img_path = get_image_path("dog.jpeg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Create meshgrid
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    X, Y = np.meshgrid(x, y)

    # Apply 5x5 Gaussian Blur
    blur_5x5 = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply 11x11 Gaussian Blur
    blur_11x11 = cv2.GaussianBlur(img, (11, 11), 0)

    display_images(X, Y, img, blur_5x5, blur_11x11)

    # Answer to the question:
    # As the filter size increases, the 3D graphs become smoother, and the sharp variations in pixel intensity are reduced. 
    # The image surface appears more flattened, indicating a loss of fine details due to increased blurring.

def display_images(
    x: np.ndarray,
    y: np.ndarray,
    org_img: np.ndarray,
    blur_5x5: np.ndarray,
    blur_11x11: np.ndarray,
) -> None:
    """
    Display the original, 5x5 blurred, and 11x11 blurred 3D in a 1x3 meshgrid.
    Args:
        org_img (np.ndarray): The original image.
        blur_5x5 (np.ndarray): The 5x5 blurred image.
        blur_11x11 (np.ndarray): The 11x11 blurred image
    """

    # Display the images in a 1x3 grid
    _, axes = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={"projection": "3d"})
    axes: list[Axes3D] # Type hint for axes

    axes[0].plot_surface(X=x, Y=y, Z=org_img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].plot_surface(X=x, Y=y, Z=blur_5x5, cmap="gray")
    axes[1].set_title("5x5 Gaussian Blur")
    axes[1].axis("off")

    axes[2].plot_surface(X=x, Y=y, Z=blur_11x11, cmap="gray")
    axes[2].set_title("11x11 Gaussian Blur")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
