# Author: Sukhrob Ilyosbekov
# Date: 10/06/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core import get_image_path, scale_image

def main() -> None:
    """
    Apply a low-pass and high-pass filter to two images and combine
    them to create a hybrid image.
    """

    # Get the image path from the dataset/images directory
    musk_path = get_image_path("musk.jpg")
    trump_path = get_image_path("trump.jpg")
    musk_img = cv2.imread(musk_path, cv2.IMREAD_GRAYSCALE)
    trump_img = cv2.imread(trump_path, cv2.IMREAD_GRAYSCALE)

    # Resize the images to either width or height of 512
    musk_img = scale_image(musk_img, 512, 512)
    trump_img = scale_image(trump_img, 512, 512)

    low_pass_img = low_pass_filter(musk_img)
    high_pass_img = high_pass_filter(trump_img)
    combined_img = combine_images(low_pass_img, high_pass_img)

    display_images(low_pass_img, high_pass_img, combined_img)

def low_pass_filter(image: np.ndarray, kernel_size = 11) -> np.ndarray:
    """
    Apply a simple low-pass filter to an image.
    Args:
        image (np.ndarray): The input image.
        kernel_size (int): The size of the filter kernel. Default is 11.
    Returns:
        np.ndarray: The low-pass filtered image.
    """
    # Create a kernel with all values equal to 1/(kernel_size * kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    image_height, image_width = image.shape
    pad_size = kernel_size // 2

    # Pad the image with reflection
    padded_image = np.pad(image, pad_size, mode="reflect")
    low_pass_image = np.zeros_like(image, dtype=np.float32)
    
    # Apply the low-pass filter using convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            low_pass_image[i, j] = np.sum(region * kernel)
    
    return low_pass_image

def high_pass_filter(image: np.ndarray, kernel_size = 11) -> np.ndarray:
    """
    Apply a simple high-pass filter to an image by subtracting the low-pass filtered image and adding 127.
    Args:
        image (np.ndarray): The input image.
        kernel_size (int): The size of the filter kernel. Default is 11.
    Returns:
        np.ndarray: The high-pass filtered image.
    """
    low_pass_image = low_pass_filter(image, kernel_size)
    high_pass_image = image - low_pass_image
    high_pass_image = high_pass_image + 127
    return high_pass_image

def combine_images(low_pass_img: np.ndarray, high_pass_img: np.ndarray) -> np.ndarray:
    """
    Combine the low-pass and high-pass filtered images.
    Args:
        low_pass_img (np.ndarray): The low-pass filtered image.
        high_pass_img (np.ndarray): The high-pass filtered image.
    Returns:
        np.ndarray: The combined image.
    """

    # Hidden step: normalize the pixel values to the range [0, 255]
    # and adjust alpha values for each image to create a better hybrid image
    norm_low_pass_img = cv2.normalize(low_pass_img, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    norm_high_pass_img = cv2.normalize(high_pass_img, None, 0, 255, cv2.NORM_MINMAX) # type: ignore

    # 30% of low-pass image and 70% of high-pass image
    combined_image = cv2.addWeighted(norm_low_pass_img, 0.3, norm_high_pass_img, 0.7, 0)
    return combined_image

def display_images(low_pass_img: np.ndarray, high_pass_img: np.ndarray, combined_img: np.ndarray) -> None:
    """
    Display the original image, low-pass filtered image, high-pass filtered image, and combined image.
    Args:
        low_pass_img (np.ndarray): The low-pass filtered image.
        high_pass_img (np.ndarray): The high-pass filtered image.
        combined_img (np.ndarray): The combined image.
    """
    # Display the images in a 1x3 grid
    _, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes: list[Axes] # Type hint for axes

    axes[0].imshow(low_pass_img, cmap="gray")
    axes[0].set_title("Low-Pass Filtered Image")
    axes[0].axis("off")

    axes[1].imshow(high_pass_img, cmap="gray")
    axes[1].set_title("High-Pass Filtered Image")
    axes[1].axis("off")

    axes[2].imshow(combined_img, cmap="gray")
    axes[2].set_title("Combined Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
