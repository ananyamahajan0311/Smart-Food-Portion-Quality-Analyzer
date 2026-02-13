import cv2
import numpy as np

def segment_food(original_img):
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu Thresholding
    _, mask = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        return clean_mask

    return mask
