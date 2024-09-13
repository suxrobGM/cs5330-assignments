# Author: Sukhrob Ilyosbekov
# Date: 09/13/2024
# Description: This file is the main entry point for the lab1 project.

import cv2
from core import get_image_path

def main() -> None:
    """
    Display the original and grayscaled images of flowers.jpg image in separate windows.
    """

    # Get the image path from the dataset/images directory
    imag_path = get_image_path("flowers.jpg")

    # Read the original
    original_image = cv2.imread(imag_path)

    # Read the grayscaled image
    grayscaled_image = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)

    # Display original and grayscaled images
    cv2.imshow("Original Flowers", original_image)
    cv2.imshow("Grayscaled Flowers", grayscaled_image)
    cv2.waitKey(0) # Prevent the window from closing immediately
