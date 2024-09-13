import os

def get_image_path(image_name: str) -> str:
    """
    Get the full path of the image in the dataset/images directory.
    """
    return os.path.join(os.path.dirname(__file__), "../../dataset/images", image_name)
