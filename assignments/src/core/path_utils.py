import os

def get_image_path(image_file: str) -> str:
    """
    Get the full path of the image in the dataset/images directory.
    Args:
        image_file (str): The name of the image file including the extension.
    Returns:
        str: The full path of the image file.
    Raises:
        FileNotFoundError: If the image file is not found.
    """
    image_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../dataset/images", image_file))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_file}")
    
    return image_path
