import cv2
import numpy as np
import tempfile


def ela_score(path, quality=90):

    img = cv2.imread(path)

    if img is None:
        return None

    # save temporary JPEG
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

    cv2.imwrite(temp_file.name, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    recompressed = cv2.imread(temp_file.name)

    diff = cv2.absdiff(img, recompressed)

    return np.mean(diff)