import cv2
import numpy as np
from core.loader import load_image

def laplacian_variance(path):
    img = load_image(path) 

    img_uint8 = (img * 255).astype(np.uint8)

    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    variance = laplacian.var()

    return variance