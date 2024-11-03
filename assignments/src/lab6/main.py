# Author: Sukhrob Ilyosbekov
# Date: 10/22/2024

import cv2
import numpy as np
from core import get_image_path

def main():
    """
    Detect contours of colored blocks in an image using HSV color space.
    """
    # Load the image
    image_path = get_image_path("blocks.jpg")
    image = cv2.imread(image_path)

    # Convert the image to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lower bound of saturation and value for better color detection
    lower_saturation = 100
    lower_value = 50

    # We define the color ranges lower and upper bounds in HSV color space to detect colored blocks

    # Red color range HSV
    # Red color has a hue value of 0, but it is split into two ranges due to the circular nature of the HSV color space
    # So, we need to create two masks for red color and combine them
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
    # after applying the masks, we will have a binary image with the detected colors as white and the rest as black
    mask = cv2.bitwise_or(mask_red, mask_green)
    mask = cv2.bitwise_or(mask, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_brown)
    mask = cv2.bitwise_or(mask, mask_orange)

    # Apply morphological operations to clean up the mask
    # It makes the mask more robust and removes noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours on the mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and find center points
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > 500:
            # Draw contour
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            # Calculate moments to find the center point
            # m00 is the area of the contour
            # m10 and m01 are the moments of the contour which means the weighted average of the x and y coordinates
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw center point as a blue circle
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

                # Draw the coordinates of the center point
                cv2.putText(image, f"({cx}, {cy})", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display masked image, cleaned mask, and contours with center points
    cv2.namedWindow("Mask", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Cleaned Mask", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Contours and Centers", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Mask", 1000, 659)
    cv2.resizeWindow("Cleaned Mask", 1000, 659)
    cv2.resizeWindow("Contours and Centers", 1000, 659)

    cv2.imshow("Mask", mask)
    cv2.imshow("Cleaned Mask", cleaned_mask)
    cv2.imshow("Contours and Centers", image)
    capture_hsv_values(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture_hsv_values(image: np.ndarray) -> None:
    """
    Capture HSV values of pixels in an image using mouse events and display them in the console.
    Args:
        image (np.ndarray): The input image.
    """
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
