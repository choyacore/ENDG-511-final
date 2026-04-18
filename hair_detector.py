
"""
hair_detector.py - Face detection and hair region extraction.


Data flow: frame_bgr -> FaceHairDetector
.detect() -> FaceHairRegion

           -> hair_box goes to color_classifier.py
           -> full_head_box + forehead_y + chin_y go to length_classifier.py

NOTE: Originally tried MediaPipe FaceMesh for landmark-based forehead detection
but it doesn't support Python 3.12+ on Windows without errors .
Switched to Haar Cascade -- less precise on angled faces but runs on all team setups.
See: https://github.com/google/mediapipe/issues/3400
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# TODO: currently misses faces that arent close to camera



FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

"""
        Convert a raw face rect into the set of boxes that color and length
        classifiers actually need.
 
        hair_box geometry:
          - extends _HAIR_TOP_RATIO * face_height above the forehead
          - padded horizontally by _SIDE_PAD_RATIO * face_width on each side
            because hair is always wider than the face rect the cascade returns
          - height is capped at forehead_y so it never dips into the face region
            and accidentally samples eyebrow or skin pixels
 
        full_head_box geometry:
          - same horizontal bounds as hair_box
          - y starts at 0 (top of frame) not at hair_top -- length_classifier
            needs to measure how far hair falls *below* the chin too, so a
            cropped strip misses long hair sitting on shoulders
"""
@dataclass
class FaceHairRegion:
    face_box:      tuple  # (x,y,w,h) raw cascade output -- used for debug overlay drawing
    hair_box:      tuple    # strip above forehead, fed into color_classifier.classify_color()
    full_head_box: tuple    # height capped at forehead_y so it never overlaps the face rect
    # full vertical strip y=0 to frame bottom -- fed into length_classifier
    # starts at y=0 not hair_top so shoulder-length hair isn't clipped
    forehead_y:    int  # top edge of face
    chin_y:        int # bottom edge of face
    approx_landmark:  np.ndarray   # just face corners , enough for debug box


# TODO: 0.6 hair_top ratio clips very tall hairstyles (e.g. high buns)
# Sehba flagged this during length classifier testing -- fine for demo
# but worth revisiting if we expand the dataset
#research what happens for bald people (was asked during the presentation)


class FaceHairDetector:

    def __init__(self, max_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        # params kept for API compatibility in case we swap back to MediaPipe
        # Haar Cascade doesn't use any of these -- only self._cascade matters
        self._cascade = FACE_CASCADE

    def detect(self, frame_bgr):
        # Haar Cascade requires single-channel input , colour info not used for detection
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        # largest face by area = closest to camera = the user, ignore background faces
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return self.extract_hair_region(x, y, w, h, frame_bgr.shape[:2])
    
    def extract_hair_region(self, x, y, w, h, frame_shape):
        fh, fw     = frame_shape
        face_box   = (int(x), int(y), int(w), int(h))
        forehead_y = int(y)
        chin_y     = int(y + h)
        pad_x      = int(w * 0.20)
        hx_min     = max(0, x - pad_x)
        hx_max     = min(fw, x + w + pad_x)
        hair_top   = max(0, y - int(h * 0.6))# lower values that 0.6 clipped top of hair on longer styles
        hair_box      = (int(hx_min), int(hair_top),
                         int(hx_max - hx_min),
                         max(0, forehead_y - hair_top))
        full_head_box = (int(hx_min), 0, int(hx_max - hx_min), fh)
        approx_landmark  = np.array([[x, y], [x+w, y+h]], dtype=np.int32)
        return FaceHairRegion(
            face_box=face_box, hair_box=hair_box,
            full_head_box=full_head_box, forehead_y=forehead_y,
            chin_y=chin_y, approx_landmark=approx_landmark,
        )

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()