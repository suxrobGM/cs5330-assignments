import cv2
import tkinter as tk
import numpy as np
from threading import Thread
from tkinter import Button, Label, StringVar, ttk
from PIL import Image, ImageTk
from deepface import DeepFace
from core.utils import format_deepface_results

class WebcamTab(tk.Frame):
    camera_var: StringVar
    camera_dropdown: ttk.Combobox
    video_label: Label
    result_label: Label
    capture: cv2.VideoCapture | None
    available_cameras: list[int]
    current_image: ImageTk.PhotoImage
    frame_skip_count: int = 0
    process_every_nth_frame: int = 1  # Only detect faces every 5th frame

    def __init__(self, parent: tk.Tk) -> None:
        tk.Frame.__init__(self, parent)
        self.parent = parent

        # Dropdown for camera selection
        self.camera_var = StringVar()
        self.camera_dropdown = ttk.Combobox(self, textvariable=self.camera_var)
        self.camera_dropdown.grid(row=0, column=0, padx=10, pady=10)
        self.camera_dropdown["state"] = "readonly"

        # Button to start the webcam
        start_button = Button(self, text="Start Camera", command=self.start_camera)
        start_button.grid(row=0, column=1, padx=10, pady=10)

        stop_button = Button(self, text="Stop Camera", command=self.stop_camera)
        stop_button.grid(row=0, column=2, padx=10, pady=10)

        # Label to display the webcam feed
        self.video_label = Label(self)
        self.video_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Label to display the results
        self.result_label = Label(self, text="", wraplength=300)
        self.result_label.grid(row=2, column=0, columnspan=2)

        # Initialize attributes
        self.capture = None
        self.available_cameras = self.get_available_cameras()
        self.populate_camera_dropdown()

    def get_available_cameras(self) -> list:
        """
        Scans and returns a list of available camera indices.
        """
        available_cameras = []
        for i in range(5):  # Checking the first 5 indexes for available cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def populate_camera_dropdown(self) -> None:
        """
        Populates the camera dropdown with available camera indices.
        """
        camera_options = [(f"Camera {i}") for i in self.available_cameras]
        self.camera_dropdown["values"] = camera_options
        if camera_options:
            self.camera_dropdown.current(0)  # Select the first camera by default

    def start_camera(self) -> None:
        """
        Starts the webcam feed using the selected camera.
        """
        # Get the selected camera index from the dropdown
        selected_camera = int(self.camera_dropdown.get().split()[-1])

        print(f"Starting camera {selected_camera}")

        # If there's already an active thread, stop it before starting a new one
        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(selected_camera)  # Open the selected webcam
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        Thread(target=self.show_webcam).start()

    def show_webcam(self) -> None:
        while True and self.capture is not None:
            ret, frame = self.capture.read()
            if not ret:
                return

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze the frame in a separate thread
            # Skip frames for face detection to reduce lag
            #self.frame_skip_count += 1
            #if self.frame_skip_count % self.process_every_nth_frame == 0:
                # Run face detection on a resized copy for faster processing
                # small_frame = cv2.resize(rgb_frame, (320, 240))
                #Thread(target=self.recognize_face_from_frame, args=(rgb_frame, frame)).start()

            self.recognize_face_from_frame(rgb_frame, frame)

            # Display the frame in the GUI
            self.display_frame(rgb_frame)

    def recognize_face_from_frame(self, rgb_frame: np.ndarray, original_frame: np.ndarray) -> None:
        try:
            # Perform face recognition using DeepFace
            results = DeepFace.extract_faces(original_frame, enforce_detection=False)
            face_boxes = []

            for result in results:
                # Draw rectangles on the original frame
                face_box = result.get("facial_area")

                if face_box:
                    x, y, w, h = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
                    # scale_x = original_frame.shape[1] / rgb_frame.shape[1]
                    # scale_y = original_frame.shape[0] / rgb_frame.shape[0]

                    # x, y, w, h = (
                    #     int(face_box["x"] * scale_x),
                    #     int(face_box["y"] * scale_y),
                    #     int(face_box["w"] * scale_x),
                    #     int(face_box["h"] * scale_y),
                    # )

                    face_boxes.append((x, y, w, h))

                    # Schedule rectangle drawing in the main thread using after()
                    # self.parent.after(0, self.draw_rectangles, rgb_frame, face_boxes)
                    self.draw_rectangles(rgb_frame, face_boxes)

            # Format and display the results
            #formatted_results = format_deepface_results(results)
            # self.result_label.config(text=formatted_results)

        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")

    def draw_rectangles(self, frame: np.ndarray, face_boxes: list[tuple[int, int, int, int]]) -> None:
        """
        Draws rectangles on the frame for all detected faces and updates the GUI.
        Args:
            frame (np.ndarray): The RGB frame to draw rectangles on.
            face_boxes (list[tuple[int, int, int, int]]): A list of face bounding boxes.
        """
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update the GUI with the modified frame
        self.display_frame(frame)

    def display_frame(self, frame: np.ndarray) -> None:
        """
        Displays the frame in the GUI.
        Args:
            frame (np.ndarray): The RGB frame to display.
        """
        image = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_label.configure(image=image) # type: ignore
        self.video_label.image = image # type: ignore

    def stop_camera(self) -> None:
        """
        Stops the webcam feed.
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.video_label.config(image=None) # type: ignore
            self.result_label.config(text="")
