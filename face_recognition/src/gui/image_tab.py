from tkinter import Button, Frame, Tk, filedialog, Label
from PIL import Image, ImageTk, ImageDraw
from deepface import DeepFace
from threading import Thread

class ImageTab(Frame):
    upload_button: Button
    uploaded_img_label: Label
    result_img_label: Label
    result_label: Label

    def __init__(self, parent: Tk) -> None:
        super().__init__(parent)
        self.parent = parent

        Label(self, text="Upload an image to recognize faces").grid(row=0, column=0)

        self.upload_button = Button(self, text="Select an image", command=self.upload_image)
        self.upload_button.grid(row=1, column=0)
        
        self.uploaded_img_label = Label(self)
        self.uploaded_img_label.grid(row=2, column=0, pady=10)
        
        self.result_img_label = Label(self)
        self.result_img_label.grid(row=2, column=1, pady=10)

        self.result_label = Label(self, text="", wraplength=200)
        self.result_label.grid(row=3, column=0)

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path: str) -> None:
        image = Image.open(file_path)
        self.set_image_label(self.uploaded_img_label, image, resize=(400, 400))
        self.result_label.config(text="Recognizing faces...")
        Thread(target=lambda: self.recognize_face(file_path)).start()

    def recognize_face(self, file_path: str) -> None:
        """
        Recognize faces in an image using DeepFace.
        Args:
            file_path (str): The path to the image file.
        """
        try:
            results = DeepFace.analyze(file_path, actions=["age", "gender", "race", "emotion"], enforce_detection=False)

            formatted_results = self.format_deepface_results(results)
            self.result_label.config(text=formatted_results)

            self.draw_rectangles_on_faces(file_path, results)
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")

    def draw_rectangles_on_faces(self, file_path: str, results: list[dict]) -> None:
        """
        Draw rectangles around the detected faces and display the updated image.
        Args:
            file_path (str): The path to the image file.
            results (list): The list of DeepFace results with bounding box information.
        """
        # Open the original image
        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)

        # Loop through each face result and draw a rectangle
        for result in results:
            face_box = result.get("region")
            if face_box:
                x, y, w, h = face_box["x"], face_box["y"], face_box["w"], face_box["h"]
                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=3)

        self.set_image_label(self.result_img_label, image, resize=(400, 400))

    def format_deepface_results(self, results: list[dict]) -> str:
        """
        Formats the results of DeepFace analysis for multiple faces into a multiline string.
        Args:
            results (list): The result from DeepFace analysis containing an array of dicts.
        Returns:
            str: A formatted multiline string displaying the analysis for all faces.
        """
        formatted_result = "DeepFace Analysis Results:\n-----------------------------\n"
        
        for idx, result in enumerate(results):
            emotion: str = result.get("dominant_emotion", "N/A")
            age: str = result.get("age", "N/A")
            gender: str = result.get("dominant_gender", "N/A")
            race: str = result.get("dominant_race", "N/A")
            
            formatted_result += f"Face {idx + 1}:\n"
            formatted_result += f"  Emotion: {emotion.capitalize()}\n"
            formatted_result += f"  Age: {age}\n"
            formatted_result += f"  Gender: {gender}\n"
            formatted_result += f"  Race: {race.capitalize()}\n"
            formatted_result += "-----------------------------\n"
        
        return formatted_result
    
    def resize_image_to_fit(self, image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """
        Resize an image to fit within the specified maximum width and height.
        Args:
            image (Image.Image): The image to resize.
            max_width (int): The maximum width.
            max_height (int): The maximum height.
        Returns:
            Image.Image: The resized image.
        """
        width, height = image.size

        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height))
        
        return image
    
    def set_image_label(
        self,
        label: Label,
        image: Image.Image,
        resize: tuple[int, int] | None = None
    ) -> None:
        """
        Set the image label to display the given image.
        Args:
            label (Label): The label to set the image to.
            image (Image): PIL Image to display.
            resize (tuple[int, int]): The size to resize the image to. The first element is the width, the second is the height. Default is None.
        """
        resized_image = image

        if resize:
            resized_image = self.resize_image_to_fit(image, max_width=resize[0], max_height=resize[1])
        
        photo_image = ImageTk.PhotoImage(resized_image)
        label.configure(image=photo_image) # type: ignore
        label.image = photo_image # type: ignore

