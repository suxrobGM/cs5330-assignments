import numpy as np
import math

def main() -> None:
    # Define the 3x3 pixel grid
    pixel_grid = np.array([
        [0, 255, 255],
        [0, 0, 255],
        [0, 0, 0]
    ])

    # Sobel filters
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Convolve the grid with Sobel filters
    G_x = np.sum(sobel_x * pixel_grid)
    G_y = np.sum(sobel_y * pixel_grid)

    # Edge normal (magnitude of the gradient)
    magnitude = math.sqrt(G_x**2 + G_y**2)

    # Edge direction (angle of the gradient)
    theta = math.degrees(math.atan2(G_y, G_x))

    print(f"G_x: {G_x}")
    print(f"G_y: {G_y}")
    print(f"Magnitude: {magnitude}")
    print(f"Theta: {theta}")

