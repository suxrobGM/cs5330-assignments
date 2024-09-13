import os

def get_image_path(image_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "../../dataset/images", image_name)
