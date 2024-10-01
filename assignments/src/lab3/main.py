# Author: Sukhrob Ilyosbekov
# Date: 09/29/2024
# Description: This file is the main entry point for the lab3 project.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import get_image_path

def main() -> None:
    """
    Apply salt and pepper noise to an image and filter it using a neighborhood filter.
    """

    # Get the image path from the dataset/images directory
    imag_path = get_image_path("dog.jpeg")
    image = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)

    # Add salt and pepper noise to the image for 1%, 10%, and 50% noise levels
    noisy_1 = add_salt_pepper_noise(image, 0.01)
    noisy_10 = add_salt_pepper_noise(image, 0.10)
    noisy_50 = add_salt_pepper_noise(image, 0.50)

    # Apply a 3x3 neighborhood filter to the noisy images
    filtered_1_3x3 = apply_neighborhood_filter_optimized(noisy_1, 3)
    filtered_10_3x3 = apply_neighborhood_filter_optimized(noisy_10, 3)
    filtered_50_3x3 = apply_neighborhood_filter_optimized(noisy_50, 3)

    # Apply a 5x5 neighborhood filter to the noisy images
    filtered_1_5x5 = apply_neighborhood_filter_optimized(noisy_1, 5)
    filtered_10_5x5 = apply_neighborhood_filter_optimized(noisy_10, 5)
    filtered_50_5x5 = apply_neighborhood_filter_optimized(noisy_50, 5)

    # Display the images im a 3x3 grid
    _, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Original images with noise
    axes[0, 0].imshow(noisy_1, cmap="gray")
    axes[0, 0].set_title("1% Noise")
    axes[0, 1].imshow(noisy_10, cmap="gray")
    axes[0, 1].set_title("10% Noise")
    axes[0, 2].imshow(noisy_50, cmap="gray")
    axes[0, 2].set_title("50% Noise")

    # Filtered images (3x3)
    axes[1, 0].imshow(filtered_1_3x3, cmap="gray")
    axes[1, 0].set_title("3x3 Filter on 1%")
    axes[1, 1].imshow(filtered_10_3x3, cmap="gray")
    axes[1, 1].set_title("3x3 Filter on 10%")
    axes[1, 2].imshow(filtered_50_3x3, cmap="gray")
    axes[1, 2].set_title("3x3 Filter on 50%")

    # Filtered images (5x5)
    axes[2, 0].imshow(filtered_1_5x5, cmap="gray")
    axes[2, 0].set_title("5x5 Filter on 1%")
    axes[2, 1].imshow(filtered_10_5x5, cmap="gray")
    axes[2, 1].set_title("5x5 Filter on 10%")
    axes[2, 2].imshow(filtered_50_5x5, cmap="gray")
    axes[2, 2].set_title("5x5 Filter on 50%")

    plt.tight_layout()
    plt.show()

    
def add_salt_pepper_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add salt and pepper noise to the image.
    Args:
        image (np.ndarray): The input image.
        noise_level (float): The noise level.
    Returns:
        np.ndarray: The image with salt and pepper noise.
    """

    noisy_image = np.copy(image)
    num_salt = int(noise_level * image.size // 2) # Number of salt noise pixels
    num_pepper = int(noise_level * image.size // 2)
    
    # Add salt noise (white)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords_salt[0], coords_salt[1]] = 255
    
    # Add pepper noise (black)
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords_pepper[0], coords_pepper[1]] = 0
    return noisy_image


def apply_neighborhood_filter(image: np.ndarray, filter_size: int) -> np.ndarray:
    """
    Apply a neighborhood filter to the image.
    Args:
        image (np.ndarray): The input image.
        filter_size (int): The filter size.
    Returns:
        np.ndarray: The filtered image.
    """
    
    # Pad the image with zeros
    padded_image = np.pad(image, pad_width=filter_size, mode="constant", constant_values=0)

    # Create an empty image to store the filtered image
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Apply filter to the neighborhood
            neighborhood = padded_image[i:i + filter_size, j:j + filter_size]
            filtered_image[i, j] = np.mean(neighborhood)
    
    return filtered_image

def apply_neighborhood_filter_optimized(image: np.ndarray, filter_size: int) -> np.ndarray:
    """
    Optimized version of the apply_neighborhood_filter function.
    It uses numpy vectorized operations to apply the filter.
    Args:
        image (np.ndarray): The input image.
        filter_size (int): The filter size.
    Returns:
        np.ndarray: The filtered image.
    """
    
    # Pad the image with zeros
    padded_image = np.pad(image, pad_width=filter_size, mode="constant", constant_values=0)

    # Shape of the sliding windows view, it will be 4D (2 spatial dimensions + 2 filter dimensions) array
    shape = (image.shape[0], image.shape[1], filter_size, filter_size) 

    # Strides of the sliding windows view, strides means how many bytes to jump to get to the next element
    strides = padded_image.strides + padded_image.strides

    # Create a sliding windows view of the padded image
    sliding_windows = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    
    # Compute the mean over the neighborhood dimensions (axis 2 and 3)
    filtered_image = np.mean(sliding_windows, axis=(2, 3))
    return filtered_image

