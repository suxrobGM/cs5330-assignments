import cv2
from core import get_image_path

def main() -> None:
    imag_path = get_image_path("flowers.jpg")
    img = cv2.imread(imag_path, cv2.IMREAD_COLOR)

    # Split the image into its channels
    b, g, r = cv2.split(img)

    # Equalize the histogram of each channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge the equalized channels
    img_eq = cv2.merge((b_eq, g_eq, r_eq))

    cv2.imshow("Original image", img)
    cv2.imshow("Equalized image", img_eq)
    cv2.waitKey(0)
