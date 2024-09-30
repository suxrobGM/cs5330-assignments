# Author: Sukhrob Ilyosbekov
# Date: 09/22/2024
# Description: This file is the main entry point for the lab2 project.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import get_image_path

def main() -> None:
    """
    Display grayscale images using different methods.
    """

    # Get the image path from the dataset/images directory
    imag_path = get_image_path("flowers.jpg")
    original_image = cv2.imread(imag_path)

    # Convert the image to RGB to display with matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    gray_image_cv2 = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)
    gray_image_average = grayscale_average(original_image)
    gray_image_ntsc = grayscale_ntsc(original_image)

    # Display all images in a single plot
    # Create a plot with 2 rows and 2 columns for the images
    plt.figure(figsize=(10, 10))

    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Plot Average Method grayscale image
    plt.subplot(2, 2, 2)
    plt.imshow(gray_image_average, cmap="gray")
    plt.title("Average Method Grayscale")
    plt.axis("off")

    # Plot NTSC Method grayscale image
    plt.subplot(2, 2, 3)
    plt.imshow(gray_image_ntsc, cmap="gray")
    plt.title("NTSC Method Grayscale")
    plt.axis("off")

    # Plot OpenCV Grayscale image
    plt.subplot(2, 2, 4)
    plt.imshow(gray_image_cv2, cmap="gray")
    plt.title("OpenCV IMREAD_GRAYSCALE Grayscale")
    plt.axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

def grayscale_average(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale using the average method.
    Args:
        image(np.ndarray): The image to convert to grayscale.
    Returns:
        np.ndarray: The grayscaled image.
    """
    height, width, channels = image.shape

    # Create a new image to store the grayscaled image
    grayscaled_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the pixel value at the current position
            pixel = image[y, x]

            average = int((int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3)
            grayscaled_image[y, x] = average

    return grayscaled_image

def grayscale_ntsc(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale using the NTSC method.
    Args:
        image(np.ndarray): The image to convert to grayscale.
    Returns:
        np.ndarray: The grayscaled image.
    """
    height, width, channels = image.shape

    # Create a new image to store the grayscaled image
    grayscaled_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]

            # Calculate the NTSC grayscale value and set it in the grayscaled image
            grayscale = int(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0])
            grayscaled_image[y, x] = grayscale

    return grayscaled_image
