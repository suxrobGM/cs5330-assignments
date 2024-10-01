import tkinter as tk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

class WebcamTab(tk.Frame):
    def __init__(self, parent: tk.Tk) -> None:
        tk.Frame.__init__(self, parent)
        self.parent = parent

