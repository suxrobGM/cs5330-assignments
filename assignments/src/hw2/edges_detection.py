# Author: Sukhrob Ilyosbekov
# Date: 10/15/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from core import get_image_path

def run() -> None:
    # Get the image path from the dataset/images directory and load the image
    img_path = get_image_path("dog.jpeg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel edge detection
    magnitude, _ = apply_sobel(img)

    # Apply thresholding to the magnitude image
    sobel_50 = threshold(magnitude, 50)
    sobel_150 = threshold(magnitude, 150)

    # Apply Canny edge detection on the original image
    canny_edges = cv2.Canny(img, 100, 100)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Sobel on Blurred Image
    magnitude_blur, _ = apply_sobel(blurred)

    sobel_50_blur = threshold(magnitude_blur, 50)
    sobel_150_blur = threshold(magnitude_blur, 150)

    # Canny on Blurred Image
    canny_blur = cv2.Canny(blurred, 100, 100)

    titles = ["Original", "Sobel cut_off 50", "Sobel cut_off 150", "Canny",
              "Blurred", "", "", ""]

    images = [img, sobel_50, sobel_150, canny_edges,
              blurred, sobel_50_blur, sobel_150_blur, canny_blur]

    display_images(images, titles)

    # Answer to questions:
    # - What did you notice when you went from a lower threshold value to a higher one?
    #   Increasing the threshold value reduces the number of edges detected.
    #   With a higher threshold, only the strongest edges are kept, resulting in a cleaner but less detailed edge map.
    #   Conversely, a lower threshold retains more edges, including weaker ones, which may include noise.

    # - What did you notice before and after applying a Gaussian Blur to the image?
    #   Applying Gaussian Blur before edge detection reduces noise and smooths the image,
    #   leading to more continuous and clearer edges.
    #   The edge maps after blurring have fewer spurious edges, and the detected edges are less jagged.

def display_images(images: list[np.ndarray], titles: list[str]) -> None:
    """
    Display a list of images with corresponding titles.
    Args:
        images (list[np.ndarray]): A list of images to display.
        titles (list[str]): A list of titles for the images.
    """
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 4, i)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()


def apply_sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Sobel edge detection to an image without using any OpenCV functions.
    Args:
        img (np.ndarray): The input image as a grayscale image.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the magnitude and angle of the gradients.
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Gradient in x-direction
    grad_x = convolve_2d(image, sobel_x)

    # Gradient in y-direction
    grad_y = convolve_2d(image, sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.astype(np.uint8)

    angle = np.arctan2(grad_y, grad_x)
    angle = angle * 180 / np.pi  # Convert to degrees
    angle[angle < 0] += 180

    return magnitude, angle


def threshold(image: np.ndarray, value: int) -> np.ndarray:
    """
    Apply a threshold to an image.
    Threshold the image such that all pixel values below the threshold are set to 0,
    and all pixel values above the threshold are set to 255.
    Args:
        image (np.ndarray): The input image.
        value (int): The threshold value.
    Returns:
        np.ndarray: The thresholded image.
    """
    output = np.zeros_like(image)
    output[image >= value] = 255
    return output

def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution to an image using a kernel.
    Used `as_strided` to create a view of sliding windows and `einsum` to perform the convolution which is faster than nested loops.
    Args:
        image (np.ndarray): The input image.
        kernel (np.ndarray): The convolution kernel.
    Returns:
        np.ndarray: The convolved image.
    """

    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Initialize the output image
    output = np.zeros_like(image)

    # Get the kernel size and image size
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Calculate the padding size (half of the kernel size)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))

    # Define the shape of the sliding windows (windows should match the kernel shape)
    output_shape = (image_height, image_width, kernel_height, kernel_width)
    
    # Define the strides (how to move the window)
    strides = padded_image.strides
    sliding_strides = (strides[0], strides[1], strides[0], strides[1])
    
    # Create a view of sliding windows using as_strided
    sliding_windows = as_strided(padded_image, shape=output_shape, strides=sliding_strides)

    # Perform the convolution by multiplying sliding windows by the kernel and summing
    output = np.einsum("ijkl,kl->ij", sliding_windows, kernel)
    return output
