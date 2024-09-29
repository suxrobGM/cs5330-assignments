import tkinter as tk
from tkinter import ttk
from .image_tab import ImageTab

def run() -> None:
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Face Recognition")

    image_tab = ImageTab(root)
    image_tab.initUI()
    notebook = ttk.Notebook(root)
    notebook.add(image_tab, text="Upload Image")
    notebook.pack(expand=True, fill="both")
    root.mainloop()
