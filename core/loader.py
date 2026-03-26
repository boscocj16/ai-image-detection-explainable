import cv2
import numpy as np

def load_image(path, size=(256, 256)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found or path is wrong")

    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0

    return norm