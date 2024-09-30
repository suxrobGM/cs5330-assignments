import cv2
import numpy as np

def main() -> None:
    dark_img = np.random.randint(50, size=(500,500), dtype=np.uint8)
    light_img = np.random.randint(low=205, high=255, size=(500,500), dtype=np.uint8)
    
    # Concatenate both dark and light images horizontally
    dark_light = np.concatenate((dark_img, light_img), axis=1)
    cv2.imshow("Original Image", dark_light)
    
    # Apply adaptive histogram equalization (CLAHE) to the entire image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_img = clahe.apply(dark_light)
    
    # Show the equalized image
    cv2.imshow("Adaptive Histogram Equalization", equalized_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
