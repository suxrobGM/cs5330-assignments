import cv2
from core import get_image_path

def main():
    imag_path = get_image_path("testimage1.png")
    img = cv2.imread(imag_path)
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
