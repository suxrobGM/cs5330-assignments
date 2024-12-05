# Face Recognition app

Python desktop application for face recognition using [OpenCV](https://github.com/opencv/opencv) and [DeepFace](https://github.com/opencv/opencv).

## Installation
- If you have Poetry installed, you can run `poetry install` to install the dependencies.
- If you don't have Poetry installed, you can run `pip install -r requirements.txt` to install the dependencies. It's recommended to install the dependencies in a virtual environment to avoid installing them globally. 

## Running the app
- Poetry: `poetry run python src/main.py`
- Pip: `python src/main.py`

## Features
- **Image Tab**:
    - Allow the user to upload an image.
    - Detect faces and analyze facial attributes (age, nationality, emotion, and gender).
    - Display the results within the GUI.

- **Webcam Tab**:
    - Enable real-time face detection using a webcam.
    - Display the results within the GUI.

## Screenshots
![Image Tab](./docs/screenshot_1.jpg)
