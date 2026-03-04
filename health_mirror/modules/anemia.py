import cv2
import numpy as np

def extract_eyelid_region(frame, coords):
    # Select lower eyelid landmarks
    points = [
        coords[145],
        coords[159],
        coords[23],
        coords[130]
    ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add small padding
    padding = 5
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    return frame[y_min:y_max, x_min:x_max]
def compute_anemia_score(eyelid_region):
    if eyelid_region.size == 0:
        return None

    lab = cv2.cvtColor(eyelid_region, cv2.COLOR_BGR2LAB)
    A_channel = lab[:, :, 1]

    A_mean = np.mean(A_channel)

    return A_mean