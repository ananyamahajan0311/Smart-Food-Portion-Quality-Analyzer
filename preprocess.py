import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))

    # Gaussian Blur (noise removal)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    return img, hsv
