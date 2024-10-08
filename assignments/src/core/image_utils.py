import cv2
import numpy as np

def scale_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Resize an image to fit within the specified maximum width and height.
    Args:
        image (np.ndarray): Numpy array representing the image.
        max_width (int): The maximum width.
        max_height (int): The maximum height.
    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    width, height = image.shape[1], image.shape[0]

    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))
    
    return image
