import cv2

def main() -> None:
    """
    Capture 2 video streams from the webcam and display them side by side.
    The first video stream should display the original video feed.
    The second video stream should display the video feed with the edges detected using Canny edge detection.
    """

    # Open the webcam
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frames
        _, frame = vid.read()

        # Apply Canny edge detection to the second video stream
        edges = cv2.Canny(frame, 100, 100)

        # Display the video frames
        cv2.imshow("Original", frame)
        cv2.imshow("Edges", edges)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video streams
    vid.release()
    cv2.destroyAllWindows()

