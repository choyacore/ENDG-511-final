"""
color_classifier.py – Hair colour classification using HSV and LAB colour spaces.

Pipeline
--------
1. Crop the hair ROI from the frame.
2. Remove likely skin-tone pixels (reduces false signals from forehead overlap).
3. Find the dominant colour using KMeans clustering (k=3).
4. Map the dominant HSV value to a named hair colour via lookup table.

ENDG 511 – Team 14  |  Primary owner: Sehba Samman
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import HSV_COLOR_RANGES, COLOR_FALLBACK

# ─── Constants ────────────────────────────────────────────────────────────────
N_CLUSTERS   = 3      # dominant-colour clusters
MAX_PIXELS   = 2500   # subsample limit for KMeans speed

# Skin-tone HSV mask (remove forehead bleed-in from hair ROI)
SKIN_LOWER = np.array([ 0,  25,  60], dtype=np.uint8)
SKIN_UPPER = np.array([25, 175, 255], dtype=np.uint8)

# Minimum number of non-skin pixels required to proceed
MIN_VALID_PIXELS = 60


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _crop_hair_roi(frame_bgr: np.ndarray,
                   hair_box: tuple) -> np.ndarray:
    """Return the cropped hair region (may be empty)."""
    x, y, w, h = hair_box
    if w <= 0 or h <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame_bgr[y:y + h, x:x + w]


def _remove_skin_pixels(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Mask out skin-tone pixels and return the remaining flat pixel array (BGR).
    Falls back to all pixels if fewer than MIN_VALID_PIXELS remain.
    """
    if roi_bgr.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    hsv      = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    skin_msk = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
    non_skin = roi_bgr[skin_msk == 0]

    if len(non_skin) >= MIN_VALID_PIXELS:
        return non_skin
    # Fall back: keep everything (small hair ROI, or very blonde/light hair)
    return roi_bgr.reshape(-1, 3)


def _dominant_bgr(pixels: np.ndarray) -> np.ndarray:
    """
    Cluster pixel colours with KMeans and return the BGR value of the
    largest (most frequent) cluster centroid.
    """
    if len(pixels) < N_CLUSTERS:
        return np.array([30, 30, 30], dtype=np.uint8)

    # Subsample for speed
    if len(pixels) > MAX_PIXELS:
        idx    = np.random.choice(len(pixels), MAX_PIXELS, replace=False)
        pixels = pixels[idx]

    km = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42)
    km.fit(pixels.astype(np.float32))

    counts = np.bincount(km.labels_)
    best   = counts.argmax()
    return km.cluster_centers_[best].astype(np.uint8)


def _bgr_to_hsv_pixel(bgr: np.ndarray) -> np.ndarray:
    """Convert a single (3,) BGR pixel to HSV (OpenCV convention)."""
    return cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]


def _match_hsv(hsv_pixel: np.ndarray) -> str:
    """
    Map an HSV pixel to a named hair colour using the range table in config.py.
    Checks colours in priority order: black → gray → white → brown → auburn
    → blonde → red.  Returns COLOR_FALLBACK if nothing matches.
    """
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    # Priority order matters: check dark colours first to avoid misclassification
    priority = ["black", "gray", "white", "brown", "auburn", "blonde", "red"]
    for name in priority:
        if name not in HSV_COLOR_RANGES:
            continue
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = HSV_COLOR_RANGES[name]
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return name

    return COLOR_FALLBACK


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_color(
    frame_bgr: np.ndarray,
    hair_box:  tuple,
) -> tuple[str, np.ndarray]:
    """
    Classify the hair colour from the given frame and hair bounding box.

    Parameters
    ----------
    frame_bgr : full BGR video frame
    hair_box  : (x, y, w, h) region above the forehead (from HairROI.hair_box)

    Returns
    -------
    label         : e.g. 'brown', 'blonde', 'black', …
    dominant_bgr  : (3,) uint8 BGR pixel for display (colour swatch)
    """
    roi      = _crop_hair_roi(frame_bgr, hair_box)
    pixels   = _remove_skin_pixels(roi)

    if len(pixels) == 0:
        return COLOR_FALLBACK, np.array([40, 40, 40], dtype=np.uint8)

    dominant = _dominant_bgr(pixels)
    hsv      = _bgr_to_hsv_pixel(dominant)
    label    = _match_hsv(hsv)

    return label, dominant


def get_lab_dominant(frame_bgr: np.ndarray, hair_box: tuple) -> np.ndarray:
    """
    Alternative: return the dominant colour in LAB space.
    Useful for colour-distance comparisons in the length classifier.
    """
    roi    = _crop_hair_roi(frame_bgr, hair_box)
    pixels = _remove_skin_pixels(roi)
    if len(pixels) == 0:
        return np.array([0, 128, 128], dtype=np.float32)

    dominant_bgr = _dominant_bgr(pixels)
    lab = cv2.cvtColor(
        dominant_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    return lab
