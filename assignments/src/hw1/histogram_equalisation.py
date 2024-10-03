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

    # Get images' paths
    game_of_thrones_path = get_image_path("game_of_thrones.png")
    dark_img_path = get_image_path("dark_image.jpg")

    game_of_thrones_img = cv2.imread(game_of_thrones_path, cv2.IMREAD_GRAYSCALE) # Load an image taht downloaded form internet
    dark_img = cv2.imread(dark_img_path, cv2.IMREAD_GRAYSCALE) # Load an image that was taken from Canvas

    # Equalize the histogram of images using own implementation
    dark_img_equalized_img = equalize_histogram(dark_img)
    game_of_thrones_equalized_img = equalize_histogram(game_of_thrones_img) # Equalize the histogram of the image using own implementation
    
    # Equalize the histogram of images using OpenCV
    dark_img_equalized_img_cv2 = cv2.equalizeHist(dark_img) 
    game_of_thrones_equalized_img_cv2 = cv2.equalizeHist(game_of_thrones_img) # Equalize the histogram of the image using OpenCV

    # Display orginal, equalized image, and OpenCV equalized images in a 2x3 grid
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes: list[list[Axes]] # Type hint for axes
    canvas_manager = plt.get_current_fig_manager() # Get the canvas manager

    # Set the window title of the canvas manager
    if canvas_manager:
        canvas_manager.set_window_title("Histogram Equalization")

    axes[0][0].set_title("Dark grayscaled image")
    axes[0][0].imshow(dark_img, cmap="gray")
    axes[0][0].axis("off")

    axes[0][1].set_title("Dark image histogram equalization (custom function)")
    axes[0][1].imshow(dark_img_equalized_img, cmap="gray")
    axes[0][1].axis("off")

    axes[0][2].set_title("Dark image histogram equalization (OpenCV)")
    axes[0][2].imshow(dark_img_equalized_img_cv2, cmap="gray")
    axes[0][2].axis("off")

    axes[1][0].set_title("Game of Thrones grayscaled image")
    axes[1][0].imshow(game_of_thrones_img, cmap="gray")
    axes[1][0].axis("off")

    axes[1][1].set_title("Game of Thrones histogram equalization (custom function)")
    axes[1][1].imshow(game_of_thrones_equalized_img, cmap="gray")
    axes[1][1].axis("off")

    axes[1][2].set_title("Game of Thrones histogram equalization (OpenCV)")
    axes[1][2].imshow(game_of_thrones_equalized_img_cv2, cmap="gray")
    axes[1][2].axis("off")

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
