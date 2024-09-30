# Author: Sukhrob Ilyosbekov
# Date: 09/29/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core import get_image_path

def run() -> None:
    """
    Load an image in grayscale and equalize its histogram using own implementation and OpenCV.\n
    Then, display the original, equalized image, and OpenCV equalized images in a 1x3 grid.
    """

    image_path = get_image_path("game_of_thrones.png") # Get the image path from the dataset/images directory
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load the image in grayscale
    equalized_image = equalize_histogram(image) # Equalize the histogram of the image using own implementation
    equalized_image_cv2 = cv2.equalizeHist(image) # Equalize the histogram of the image using OpenCV

    # Display orginal, equalized image, and OpenCV equalized images in a 1x3 grid
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes: tuple[Axes, Axes, Axes] # Type hint for axes
    canvas_manager = plt.get_current_fig_manager() # Get the canvas manager

    # Set the window title of the canvas manager
    if canvas_manager:
        canvas_manager.set_window_title("Histogram Equalization - Game of Thrones Image")

    axes[0].set_title("Original grayscaled image")
    axes[0].imshow(image, cmap="gray")
    axes[1].set_title("Histogram equalization using own implementation")
    axes[1].imshow(equalized_image, cmap="gray")
    axes[2].set_title("Histogram equalization using OpenCV")
    axes[2].imshow(equalized_image_cv2, cmap="gray")

    plt.tight_layout()
    plt.show()

def calc_histogram(image: np.ndarray) -> np.ndarray:
    """
    Calculate the histogram of an image with intensity values in the range [0, 255].
    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The histogram of the image.
    """

    histogram = np.zeros(256) # Create an array of zeros with 256 elements
    flatten_arr = image.ravel()  # Flatten the image array, converting it to a 1D array

    for pixel_value in flatten_arr: 
        histogram[pixel_value] += 1 # Count the number of pixels with the same intensity value
    return histogram

def calc_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative distribution function (CDF) of a histogram.
    CDF is a function that maps the intensity values to the range [0, 255].
    Args:
        histogram (np.ndarray): The input histogram.
    Returns:
        np.ndarray: The CDF of the histogram.
    """

    # Calculate the cumulative sum of the histogram
    # Cumulative sum is the sum of all elements up to the current index
    cdf = np.cumsum(histogram)

    # Normalize CDF to range between 0 and 255
    cdf_normalized: np.ndarray = ((cdf - cdf.min()) / (cdf.max() - cdf.min())) * 255
    return cdf_normalized.astype("uint8") # Convert the CDF to uint8 data type

def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of an image.
    It uses own implementation of `calc_histogram` and `calc_cdf` functions.
    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The image with equalized histogram.
    """

    histogram = calc_histogram(image) # Calculate the histogram of the image
    cdf = calc_cdf(histogram) # Calculate the CDF of the histogram

    # Apply the equalization transformation using the CDF
    equalized_image = cdf[image]
    return equalized_image
