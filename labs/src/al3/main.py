import cv2
from core import get_image_path

def main() -> None:
    imag_path = get_image_path("dog.jpeg")
    img = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)

    # Inverse image
    img = 255 - img

    cv2.imshow("Dog Image", img)
    cv2.waitKey(0)
