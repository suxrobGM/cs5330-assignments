import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

class ImageTab(tk.Frame):
    def __init__(self, parent: tk.Tk) -> None:
        tk.Frame.__init__(self, parent)
        self.parent = parent

    def initUI(self) -> None:
        self.parent.title("Face Recognition")

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=10)

        self.btn = tk.Button(self, text="Select an image", command=self.upload_image)
        self.btn.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack()

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path: str) -> None:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo_image = ImageTk.PhotoImage(Image.fromarray(image))

        self.img_label.configure(image=image) # type: ignore
        self.img_label.image = photo_image # type: ignore
        self.img_label.pack()

        self.recognize_face(file_path)

    def recognize_face(self, file_path: str) -> None:
        result = DeepFace.analyze(file_path)
        print(result)
