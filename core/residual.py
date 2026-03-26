import cv2
import numpy as np


def residual_variance(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    blur = cv2.GaussianBlur(img, (5,5), 0)

    residual = img.astype(np.float32) - blur.astype(np.float32)

    return np.var(residual)