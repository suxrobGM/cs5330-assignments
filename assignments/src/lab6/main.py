# Author: Sukhrob Ilyosbekov
# Date: 10/22/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from core import get_image_path

def main() -> None:
    """
    Convert a regular image to a night vision image.
    """

    image_path = get_image_path("blocks.jpg")
    image = cv2.imread(image_path) 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Mitigate shadows using morphological operations
    # Kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    shadow_removed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Apply a threshold to get binary image
    _, thresh = cv2.threshold(shadow_removed, 170, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and find the center points
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 500:
            # Draw contour
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            # Calculate moments to find center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw center point
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    # Display the result
    cv2.imshow("Contours and Centers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main1():
    image_path = get_image_path("blocks.jpg")
    image = cv2.imread(image_path)

    #sample_hsv_values(image)

    # Proceed with HSV conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower bound saturation and value for better color detection
    lower_saturation = 100
    lower_value = 50

    # Red color range HSV
    lower_red1 = np.array([0, lower_saturation, lower_value])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, lower_saturation, lower_value])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green color range HSV
    lower_green = np.array([40, lower_saturation, lower_value])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Blue color range HSV
    lower_blue = np.array([100, lower_saturation, lower_value])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Yellow color range HSV
    lower_yellow = np.array([20, lower_saturation, lower_value])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Brown color range HSV
    lower_brown = np.array([15, lower_saturation, 180])
    upper_brown = np.array([25, 255, 230])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Orange color range HSV
    lower_orange = np.array([10, 180, 180])
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combine the masks for all colors
    mask = cv2.bitwise_or(mask_red, mask_green)
    mask = cv2.bitwise_or(mask, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_brown)
    mask = cv2.bitwise_or(mask, mask_orange)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("Mask", mask)
    #cv2.imshow("Cleaned Mask", cleaned_mask)

    # Find contours on the mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and find center points
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 500:
            # Draw contour
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            # Calculate moments to find the center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw center point
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    # Display the result
    capture_hsv_values(image)
    #cv2.imshow("Contours and Centers", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def main2():
    image_path = get_image_path("blocks.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # apply binary thresholding
    ret, thresh = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow("Binary image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_hsv_values(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv_image[y, x]
            print(f"HSV at ({x}, {y}): {pixel}")

    cv2.namedWindow("Sample HSV")
    cv2.setMouseCallback("Sample HSV", mouse_event)
    cv2.imshow("Sample HSV", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
