import cv2
import numpy as np


def segment_food(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    height, width = image.shape[:2]
    rect = (10, 10, width-20, height-20)

    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]

    return mask2 * 255