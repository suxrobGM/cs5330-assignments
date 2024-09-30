import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core import get_image_path

def main() -> None:
    img = cv2.imread(get_image_path("unity_game.png"), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # Get x and y coordinates
    x = np.linspace(0, width, width, dtype=int)
    y = np.linspace(0, height, height, dtype=int)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection="3d")
    surf = ax.plot_surface(X, Y, img, cmap="autumn")
    plt.show()

