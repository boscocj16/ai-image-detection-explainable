import sys
import numpy as np
import joblib

from core.noise import laplacian_variance
from core.frequency import high_frequency_ratio
from core.residual import residual_variance
from core.ela import ela_score


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def interpret_noise(val):
    if val < 6:
        return "Low (AI-like)"
    elif val > 8:
        return "High (Real-like)"
    else:
        return "Moderate"


def interpret_residual(val):
    if val < 2:
        return "Low (AI-like)"
    elif val > 3:
        return "High (Real-like)"
    else:
        return "Moderate"


def interpret_ela(val):
    if val < 1:
        return "Low (AI-like)"
    elif val > 2:
        return "High (Real-like)"
    else:
        return "Moderate"


def interpret_freq(val):
    return "Weak signal (overlap)"

def detect_image(path):


    noise = laplacian_variance(path)
    freq = high_frequency_ratio(path)
    residual = residual_variance(path)
    ela = ela_score(path)


    noise_log = np.log1p(noise)
    residual_log = np.log1p(residual)
    ela_log = np.log1p(ela)


    features = np.array([[noise_log, freq, residual_log, ela_log]])


    features = scaler.transform(features)

    prob = model.predict_proba(features)[0][1]

    print("\n--- Forensic Report ---\n")

    print(f"{'Noise variance':<22}: {round(noise_log, 3)}  ({interpret_noise(noise_log)})")
    print(f"{'Frequency ratio':<22}: {round(freq, 3)}  ({interpret_freq(freq)})")
    print(f"{'Residual variance':<22}: {round(residual_log, 3)}  ({interpret_residual(residual_log)})")
    print(f"{'ELA score':<22}: {round(ela_log, 3)}  ({interpret_ela(ela_log)})")

    print("\n------------------------------")

    print(f"{'AI probability':<22}: {round(prob, 2)}")


    if prob > 0.80:
        result = "High confidence AI"
    elif prob > 0.60:
        result = "Likely AI (medium confidence)"
    elif prob < 0.20:
        result = "High confidence Real"
    elif prob < 0.40:
        result = "Likely Real (medium confidence)"
    else:
        result = "Uncertain"

    print(f"{'Prediction':<22}: {result}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
    else:
        path = " ".join(sys.argv[1:])
        detect_image(path)