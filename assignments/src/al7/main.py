import numpy as np

def main() -> None:
    matrix1 = np.array([
        [17, 19, 20],
        [121, 5, -5],
        [30, 1/5, 8]
    ])

    matrix2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    multiplication_result = np.dot(matrix1, matrix2)
    print(multiplication_result)
