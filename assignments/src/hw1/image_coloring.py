# Author: Sukhrob Ilyosbekov
# Date: 09/29/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core import get_image_path

def run() -> None:
    """
    Apply thermal coloring by converting an image to a thermal image using the brightness of the image.
    Then, display the original and thermal colored images in a 2x2 grid.
    """
    flowers_path = get_image_path("flowers.jpg", ) # Flowers image path
    telka_path = get_image_path("telka.jpg") # This image was downloaded from internet and placed in the dataset/images directory
    telka_img = cv2.imread(telka_path) # Load the telka image
    flowers_img = cv2.imread(flowers_path) # Load the flowers image

    flowers_thermal = thermal_coloring(flowers_img) # Apply thermal coloring to the flowers image
    telka_thermal = thermal_coloring(telka_img) # Apply thermal coloring to the telka image

    # Display images in a 2x2 grid
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes: tuple[tuple[Axes, Axes], tuple[Axes, Axes]] # Type hint for axes
    canvas_manager = plt.get_current_fig_manager() # Get the canvas manager

    # Set the window title of the canvas manager
    if canvas_manager:
        canvas_manager.set_window_title("Thermal Coloring")

    axes[0][0].set_title("Original flowers image")
    axes[0][0].imshow(cv2.cvtColor(flowers_img, cv2.COLOR_BGR2RGB))
    axes[0][0].axis("off")

    axes[0][1].set_title("Thermal coloring on flowers image")
    axes[0][1].imshow(cv2.cvtColor(flowers_thermal, cv2.COLOR_BGR2RGB))
    axes[0][1].axis("off")

    axes[1][0].set_title("Original telka image")
    axes[1][0].imshow(cv2.cvtColor(telka_img, cv2.COLOR_BGR2RGB))
    axes[1][0].axis("off")

    axes[1][1].set_title("Thermal coloring on telka image")
    axes[1][1].imshow(cv2.cvtColor(telka_thermal, cv2.COLOR_BGR2RGB))
    axes[1][1].axis("off")

    plt.tight_layout()
    plt.show()
    

def thermal_coloring(image: np.ndarray) -> np.ndarray:
    """
    Apply thermal coloring to an image.
    It uses the numpy vectorized operations to calculate the brightness of the image and apply the thermal coloring.
    Much faster than using nested loops.
    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The thermal colored image.
    """
    # Calculate the brightness of the image
    brightness = get_brightness(image, scale_to_one=True)

    # Initialize the output thermal image with zeros and 3 channels (RGB)
    thermal_image = np.zeros((brightness.shape[0], brightness.shape[1], 3), dtype=np.uint8)

    # Define color points for interpolation
    blue = np.array([0, 0, 255]) # Darkest (blue)
    green = np.array([127, 255, 127]) # Average (green)
    red = np.array([255, 0, 0]) # Brightest (red)

    mask_red_to_green = brightness < 0.4
    mask_green_to_blue = brightness >= 0.4

    # Interpolation between red and green for brightness < 0.4
    thermal_image[mask_red_to_green] = red + (green - red) * (brightness[mask_red_to_green, None] / 0.4)

    # Interpolation between green and blue for brightness >= 0.4
    thermal_image[mask_green_to_blue] = green + (blue - green) * ((brightness[mask_green_to_blue, None] - 0.4) / 0.6)

    return thermal_image

def get_brightness(image: np.ndarray, scale_to_one = False) -> np.ndarray:
    """
    Calculate the brightness of an image using human perception weights.
    The formula is: 0.299 * R + 0.587 * G + 0.114 * B
    Args:
        image (np.ndarray): The input image.
        scale_to_one (bool): Whether to scale the brightness to range between 0 and 1. By default, it scales to 0-255.
    Returns:
        np.ndarray: The normalized brightness of the image.
    """
    # Ensure the image has at least 3 channels
    if image.ndim < 3 and image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB)")

    # Extract the color channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Apply the brightness formula: 0.299*R + 0.587*G + 0.114*B
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Scale the brightness to [0, 1] if needed
    if scale_to_one:
        brightness = brightness / 255.0
    
    return brightness
