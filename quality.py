import cv2
import numpy as np

def analyze_quality(original_img, segmented_img):
    # Mask food region
    food = cv2.bitwise_and(original_img, original_img, mask=segmented_img)

    # Convert to grayscale
    gray = cv2.cvtColor(food, cv2.COLOR_BGR2GRAY)

    # Texture analysis using Laplacian
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color analysis
    hsv = cv2.cvtColor(food, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])

    # Simple rule-based quality classification
    if brightness > 120 and texture < 150:
        quality = "Good"
    elif brightness > 80:
        quality = "Average"
    else:
        quality = "Poor"

    return quality
