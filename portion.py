import cv2

def estimate_portion(segmented_img):
    food_pixels = cv2.countNonZero(segmented_img)

    total_pixels = segmented_img.shape[0] * segmented_img.shape[1]

    portion_percent = (food_pixels / total_pixels) * 100

    if portion_percent < 20:
        status = "Low Portion"
    elif portion_percent <= 40:
        status = "Normal Portion"
    else:
        status = "Excess Portion"

    return portion_percent, status
