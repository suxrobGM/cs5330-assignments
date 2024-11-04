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
    cap = cv2.VideoCapture(1)
    face_cascade, eye_cascade = load_cascades()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = process_frame(frame, face_cascade, eye_cascade)
        cv2.imshow("Face and Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """
    Loads Haar cascade files for face and eye detection.
    Returns:
        The face and eye Haar cascades.
    """
    face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml")
    return face_cascade, eye_cascade

def process_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier
) -> np.ndarray:
    """
    Processes each frame to detect faces and eyes, and applies an effect to invert the colors of the detected eyes.
    Args:
        frame: The frame to process.
        face_cascade: The Haar cascade for face detection.
        eye_cascade: The Haar cascade for eye detection.
    Returns:
        The processed frame.
    """
    # Convert the frame to grayscale since Haar cascades work with grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define region of interest for eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect and process eyes within each face
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Invert the eye region
            invert_eye_region(roi_color, ex, ey, ew, eh)
            
    return frame

def invert_eye_region(roi: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """
    Inverts colors in the detected eye region.
    Args:
        roi: The region of interest in the frame.
        x: The x-coordinate of the eye.
        y: The y-coordinate of the eye.
        w: The width of the eye.
        h: The height of the eye.
    """
    eye_region = roi[y:y + h, x:x + w]
    inverted_eye = cv2.bitwise_not(eye_region)
    roi[y:y + h, x:x + w] = inverted_eye

