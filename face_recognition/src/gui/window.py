import tkinter as tk
from tkinter import ttk
from .image_tab import ImageTab
from .webcam_tab import WebcamTab

def run() -> None:
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Face Recognition")
    notebook = ttk.Notebook(root)

    image_tab = ImageTab(root)
    webcam_tab = WebcamTab(root)
    
    notebook.add(image_tab, text="Upload Image")
    notebook.add(webcam_tab, text="Webcam")
    notebook.pack(expand=True, fill="both")
    root.mainloop()
