"""
length_classifier.py – Hair length classification via geometric analysis.

Strategy
--------
We know three y-positions from the face mesh:
  forehead_y  – where the hairline begins
  chin_y      – bottom of the face

We then scan downward from the chin to find the lowest row in the frame
that still contains pixels colour-similar to the detected hair colour
(using LAB colour distance for perceptual accuracy).

The length ratio is defined as:

    ratio = (lowest_hair_y - forehead_y) / (chin_y - forehead_y)

    ratio < LENGTH_SHORT_MAX  → short   (hair doesn't extend much past chin)
    ratio < LENGTH_MEDIUM_MAX → medium  (shoulder-length)
    else                      → long

ENDG 511 – Team 14  |  Primary owner: Naishah Adetunji
"""

import cv2
import numpy as np

from hair_detector import HairROI
from config import LENGTH_SHORT_MAX, LENGTH_MEDIUM_MAX

# ─── Constants ────────────────────────────────────────────────────────────────
# LAB colour distance threshold for "this pixel matches hair colour"
LAB_DISTANCE_THRESHOLD = 32.0
# Minimum number of matching pixels per row to count as a "hair row"
MIN_HAIR_PIXELS_PER_ROW = 4
# Row stride for the downward scan (larger = faster, slightly less precise)
SCAN_STRIDE = 3


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _bgr_to_lab_pixel(bgr: np.ndarray) -> np.ndarray:
    """Convert a single (3,) BGR uint8 pixel to float LAB."""
    return cv2.cvtColor(
        bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)


def _scan_for_lowest_hair_row(
    frame_bgr:    np.ndarray,
    roi:          HairROI,
    target_lab:   np.ndarray,
) -> int:
    """
    Scan downward from chin_y to the bottom of the frame.
    Return the y-coordinate of the lowest row containing at least
    MIN_HAIR_PIXELS_PER_ROW pixels whose LAB distance from target_lab
    is below LAB_DISTANCE_THRESHOLD.

    Falls back to chin_y if no matching rows are found (→ short hair).
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x, _, w, _ = roi.full_head_box
    x_lo = max(0, int(x))
    x_hi = min(frame_w, int(x + w))

    if x_hi <= x_lo:
        return roi.chin_y

    lowest_y = roi.chin_y

    for row_y in range(roi.chin_y, frame_h, SCAN_STRIDE):
        strip_bgr = frame_bgr[row_y, x_lo:x_hi]  # shape (W, 3)
        if strip_bgr.size < 3:
            continue

        # Convert strip to LAB for perceptual distance
        strip_lab = cv2.cvtColor(
            strip_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        dists = np.linalg.norm(strip_lab - target_lab, axis=1)
        if (dists < LAB_DISTANCE_THRESHOLD).sum() >= MIN_HAIR_PIXELS_PER_ROW:
            lowest_y = row_y

    return lowest_y


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_length(
    frame_bgr:    np.ndarray,
    roi:          HairROI,
    dominant_bgr: np.ndarray,
) -> tuple[str, float]:
    """
    Classify hair length using facial-landmark geometry.

    Parameters
    ----------
    frame_bgr    : full BGR frame
    roi          : HairROI from hair_detector.py
    dominant_bgr : (3,) uint8 dominant hair colour (from color_classifier.py)

    Returns
    -------
    label : 'short' | 'medium' | 'long' | 'unknown'
    ratio : float  (raw ratio for debugging / performance logging)
    """
    face_height = roi.chin_y - roi.forehead_y
    if face_height <= 0:
        return "unknown", 0.0

    target_lab  = _bgr_to_lab_pixel(dominant_bgr)
    lowest_y    = _scan_for_lowest_hair_row(frame_bgr, roi, target_lab)

    hair_extent = lowest_y - roi.forehead_y
    ratio       = hair_extent / face_height

    if ratio < LENGTH_SHORT_MAX:
        label = "short"
    elif ratio < LENGTH_MEDIUM_MAX:
        label = "medium"
    else:
        label = "long"

    return label, ratio
