import numpy as np
from core.loader import load_image


def high_frequency_ratio(path, radius_ratio=0.1):
    img = load_image(path)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2

    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow)**2 + (x - ccol)**2)

    # Define radius threshold
    max_radius = np.sqrt(crow**2 + ccol**2)
    radius_threshold = radius_ratio * max_radius

    # Low-frequency mask (inside circle)
    low_freq_mask = distance <= radius_threshold

    total_energy = np.sum(magnitude)
    low_energy = np.sum(magnitude[low_freq_mask])

    high_energy = total_energy - low_energy

    ratio = high_energy / total_energy

    return ratio