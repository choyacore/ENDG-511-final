"""
hair_detector.py – Face detection and hair ROI extraction.

Uses MediaPipe FaceMesh to detect facial landmarks, then estimates:
  - The forehead/hair boundary (top of face)
  - The chin position (for length ratio)
  - The hair bounding box (region above the forehead, within face width)

ENDG 511 – Team 14  |  Primary owner: Darren Taylor
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── MediaPipe landmark indices ───────────────────────────────────────────────
# FaceMesh provides 468 landmarks.  These indices mark the top of the forehead.
TOP_FOREHEAD_IDX = [10, 109, 67, 103, 54, 21, 162, 127, 338, 297, 332, 284, 251, 389, 356]
CHIN_IDX         = 152   # Bottom-centre of the chin


@dataclass
class HairROI:
    """
    All region-of-interest data derived from one detected face.

    Attributes
    ----------
    face_box      : (x, y, w, h) bounding box around the face landmarks
    hair_box      : (x, y, w, h) region above the forehead  — used for colour
    full_head_box : (x, y, w, h) from top-of-frame to bottom, face x-extent
                    — used to scan for hair below the chin (length estimation)
    forehead_y    : pixel y-coordinate of the forehead top
    chin_y        : pixel y-coordinate of the chin
    landmarks_px  : (N, 2) array of all landmark pixel coordinates
    """
    face_box:      tuple
    hair_box:      tuple
    full_head_box: tuple
    forehead_y:    int
    chin_y:        int
    landmarks_px:  np.ndarray


class HairDetector:
    """
    Wraps MediaPipe FaceMesh and exposes a simple detect() interface.

    Usage
    -----
    with HairDetector() as detector:
        roi = detector.detect(frame_bgr)
        if roi is not None:
            ...
    """

    def __init__(self,
                 max_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float  = 0.5):
        self._mp_fm = mp.solutions.face_mesh
        self._face_mesh = self._mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> Optional[HairROI]:
        """
        Run face-mesh detection on one BGR frame.

        Returns
        -------
        HairROI  if a face is found, otherwise None.
        """
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm  = result.multi_face_landmarks[0].landmark
        pts = np.array(
            [[int(p.x * w), int(p.y * h)] for p in lm],
            dtype=np.int32
        )  # shape (468, 2)

        return self._build_roi(pts, frame_shape=(h, w))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_roi(self, pts: np.ndarray, frame_shape: tuple) -> HairROI:
        h, w = frame_shape

        # Face bounding box
        fx_min, fy_min = pts[:, 0].min(), pts[:, 1].min()
        fx_max, fy_max = pts[:, 0].max(), pts[:, 1].max()
        face_w = fx_max - fx_min
        face_h = fy_max - fy_min
        face_box = (int(fx_min), int(fy_min), int(face_w), int(face_h))

        # Forehead y  (topmost of the top-forehead landmarks)
        forehead_y = int(pts[TOP_FOREHEAD_IDX, 1].min())
        chin_y     = int(pts[CHIN_IDX, 1])

        # Hair colour box  →  above the forehead, padded 20 % each side
        pad_x = int(face_w * 0.20)
        hx_min = max(0, fx_min - pad_x)
        hx_max = min(w, fx_max + pad_x)
        hair_box = (int(hx_min), 0, int(hx_max - hx_min), max(0, forehead_y))

        # Full-head vertical strip  →  entire frame height, same x extent
        # (used by length classifier to scan for hair below the chin)
        full_head_box = (int(hx_min), 0, int(hx_max - hx_min), h)

        return HairROI(
            face_box      = face_box,
            hair_box      = hair_box,
            full_head_box = full_head_box,
            forehead_y    = forehead_y,
            chin_y        = chin_y,
            landmarks_px  = pts,
        )

    # ── Context manager ───────────────────────────────────────────────────────

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
