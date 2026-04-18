
#TODO: Curious to see what length is detected for bald people 

import cv2
import numpy as np

from hair_detector import HairRegion
from config import short_max, medium_max



# too low missed hair rows and too high picked up background objects as hair
lab_threshold = 32.0


# 4 was the lowest value that didn't cause false positives on plain backgrounds
min_hair_pix = 4 # Minimum number of matching pixels per row to count as a "hair row"
scan_stide = 3


def bgr2labpix(bgr: np.ndarray) -> np.ndarray: #convert the dominant hair colour from color_classifier to LAB for distance comparisons in length classifier
    return cv2.cvtColor(
        bgr.reshape(1, 1, 3), cv2.color_bgr2_lab
    )[0, 0].astype(np.float32)


def scan_lowhair(
    frame_bgr:    np.ndarray,
    region:          HairRegion,
    target_lab:   np.ndarray,
) -> int:
 
    frame_h, frame_w = frame_bgr.shape[:2]
    x, _, w, _ = region.full_head_box
    x_lo = max(0, int(x))
    x_hi = min(frame_w, int(x + w))

    if x_hi <= x_lo:
        return region.chin_y

    lowest_y = region.chin_y

    for row_y in range(region.chin_y, frame_h, scan_stide):
        strip_bgr = frame_bgr[row_y, x_lo:x_hi]  # shape (W, 3)
        if strip_bgr.size < 3:
            continue


        strip_lab = cv2.cvtColor(
            strip_bgr.reshape(1, -1, 3), cv2.color_bgr2_lab
        ).reshape(-1, 3).astype(np.float32)

        dists = np.linalg.norm(strip_lab - target_lab, axis=1)
        if (dists < lab_threshold).sum() >= min_hair_pix:
            lowest_y = row_y

    return lowest_y


# TODO: length classifier struggles with updos and buns ,maybe we can add a new hair up classifier?

def classify_length(
    frame_bgr:    np.ndarray,
    region:          HairRegion,
    dominant_bgr: np.ndarray,
) -> tuple[str, float]:

    face_height = region.chin_y - region.forehead_y
    if face_height <= 0:
        return "unknown", 0.0

    target_lab  = bgr2labpix(dominant_bgr)
    lowest_y    = scan_lowhair(frame_bgr, region, target_lab)

    hair_extent = lowest_y - region.forehead_y
    ratio       = hair_extent / face_height # ratio > 1.0 means hair extends below the chin = medium/long otherwise short

    if ratio < short_max: # checking the max value for each legth type to determine the hair length
        label = "short"
    elif ratio < medium_max:
        label = "medium"
    else:
        label = "long"

    return label, ratio
