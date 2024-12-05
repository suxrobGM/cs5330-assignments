# Author: Sukhrob Ilyosbekov
# Date: 11/03/2024

import cv2
import cv2.data
import numpy as np

def main():
    """
    Detect faces and eyes in real-time using Haar cascades.
    Apply an effect to invert the colors of the detected eyes.
    """
    cap = init_video_capture(0)

    # Get the first frame
    previous_gray = get_first_frame(cap)
    blur_size = (11, 11) # Gaussian blur kernel size

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and detail in the image
        current_gray = cv2.GaussianBlur(current_gray, blur_size, 0)
        previous_gray = cv2.GaussianBlur(previous_gray, blur_size, 0)

        # Calculate the absolute difference between the current and previous frames
        frame_diff = cv2.absdiff(current_gray, previous_gray)

        # Apply thresholding to the difference frame
        thresh = apply_threshold(frame_diff)

        # Reduce noise in the thresholded image
        thresh = reduce_noise(thresh)

        # Update the previous frame
        previous_gray = current_gray

        # Find and draw contours of significant areas
        find_and_draw_contours(thresh, frame)

        cv2.imshow("Movement Detection", frame)
        cv2.imshow("Thresholded Image", thresh)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def init_video_capture(source=0) -> cv2.VideoCapture:
    """
    Initialize video capture from a webcam or video file.
    Args:
        source: Source of the video capture. Default is 0 (webcam).
    Returns:
        cap: Video capture object.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        exit()

    return cap


def get_first_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Read the first frame from the video capture.
    Args:
        cap: Video capture object.
    Returns:
        gray_frame: First frame converted to grayscale.
    """
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        exit()

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_threshold(frame_diff: np.ndarray, thresh_value=30, max_value=255) -> np.ndarray:
    """
    Apply binary thresholding to the difference image.
    Args:
        frame_diff: Difference image.
        thresh_value: Threshold value for binary thresholding. Default is 30.
        max_value: Maximum value to use with the THRESH_BINARY thresholding. Default is 255.
    Returns:
        thresh: Thresholded binary image.
    """
    _, thresh = cv2.threshold(frame_diff, thresh_value, max_value, cv2.THRESH_BINARY)
    return thresh

def reduce_noise(thresh: np.ndarray, kernel_size=(5, 5)):
    """
    Reduce noise in the thresholded image using morphological operations.
    Args:
        thresh: Thresholded binary image.
        kernel_size: Size of the structuring element. Default is (5, 5).
    Returns:
        thresh: Denoised binary image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    return thresh


def find_and_draw_contours(thresh: np.ndarray, frame: np.ndarray, min_area=750) -> None:
    """
    Find contours in the thresholded image and draw bounding boxes on the original frame.
    Multiple bounding boxes are merged into one encompassing box to track the overall movement.
    Args:
        thresh: Thresholded binary image.
        frame: Original colored frame where bounding boxes will be drawn.
        min_area: Minimum area of contours to be considered significant. Default is 750.
    """
    # Collect bounding boxes of significant contours
    bounding_boxes = []
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Append bounding boxes of significant contours to the list
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    
    if not bounding_boxes:
        return  # No significant contours found

    # Draw only the largest bounding box
    # largest_box = max(bounding_boxes, key=lambda box: box[2] * box[3])
    # x, y, w, h = largest_box
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Merging Bounding Boxes into One Encompassing Box
    x_coords = [x for (x, y, w, h) in bounding_boxes]
    y_coords = [y for (x, y, w, h) in bounding_boxes]
    w_coords = [x + w for (x, y, w, h) in bounding_boxes]
    h_coords = [y + h for (x, y, w, h) in bounding_boxes]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(w_coords)
    y_max = max(h_coords)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    