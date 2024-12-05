# Lab Description

In this lab, you will explore the `DeepFace` library to detect faces and analyze facial attributes such as nationality, age, gender, and emotion. You can learn more about `DeepFace` [here](https://github.com/serengil/deepface).  

`DeepFace` is a powerful hybrid face recognition package that integrates multiple **state-of-the-art** face recognition models, including VGG-Face, FaceNet, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, and GhostFaceNet. By default, it uses the VGG-Face model.  

The objective of this lab is to develop a Python desktop application that leverages `DeepFace` to detect faces and analyze facial attributes from an image. Additionally, you can extend the application to detect faces in real-time using a webcam (optional).  

## Lab Tasks

### 1. Install `DeepFace` and Dependencies  
   - Use a Python package manager such as `pip`, `conda`, or `poetry` to install `DeepFace` and its dependencies.  
   - **Recommendation**: Use `poetry` to manage your dependencies for better project organization. Learn more about `poetry` [here](https://python-poetry.org/).

### 2. Develop a Tkinter GUI Application  
   Create a GUI application using Tkinter with the following features:  
   - **Image Tab**:  
     - Allow the user to upload an image.  
     - Detect faces and analyze facial attributes (nationality, age, gender, and emotion).  
     - Display the results within the GUI.  
   - **Webcam Tab** (Optional):  
     - Enable real-time face detection using a webcam.  
     - Display the results within the GUI.  

### 3. Write Comments and Documentation  
   - Add clear comments to explain your code.  
   - Use Google-style docstrings for functions and classes. Learn about Google-style docstrings [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).  

### 4. Organize Your Code  
   - Avoid writing all your logic in the main file.  
   - Structure your application into well-organized functions and classes to ensure modularity and maintainability.  

This lab will help you gain practical experience in integrating advanced facial recognition models and building user-friendly desktop applications.

## Sample Output
![Sample Output](./docs/screenshot_1.jpg)
