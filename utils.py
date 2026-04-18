"""
utils.py – Visualization helpers for the Hair Analysis System.

Draws bounding boxes, info panels, colour swatches, and FPS overlay
directly onto OpenCV frames.

ENDG 511 – Team 1
"""

import cv2
import numpy as np

from hair_detector import HairROI


C_FACE    = (  0, 220,   0)   # We are using gree to box fram the face
C_HAIR    = ( 30, 165, 255)   # using orange to highlight the hair region 
C_TEXT    = (255, 255, 255)   # using white for text and labels
C_PANEL   = ( 25,  25,  25)   # dark panel background
C_ACCENT  = (100, 220, 255)   # light blue headings
C_FPS_OK  = (  0, 220,   0)   # green  – FPS ≥ 15
C_FPS_BAD = (  0,  80, 220)   # red    – FPS < 15 that means below what we planned

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FS_TITLE   = 0.70
FS_LABEL   = 0.58
FS_SMALL   = 0.44
TK_BOLD    = 2
TK_NORM    = 1

PANEL_W    = 285   # info panel width in pixels
PANEL_PAD  = 8     # margin from frame edge




def draw_roi_boxes(frame: np.ndarray, roi: HairROI) -> np.ndarray:

    out = frame.copy()

    # Face box
    fx, fy, fw, fh = roi.face_box
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), C_FACE, 2)

    # Hair colour ROI box
    hx, hy, hw, hh = roi.hair_box
    if hw > 0 and hh > 0:
        cv2.rectangle(out, (hx, hy), (hx + hw, hy + hh), C_HAIR, 2)

    # Forehead / hairline marker
    cv2.line(out, (hx, roi.forehead_y), (hx + hw, roi.forehead_y), C_HAIR, 1)
    cv2.line(out, (hx, roi.chin_y),     (hx + hw, roi.chin_y),     C_FACE, 1)

    # Labels
    cv2.putText(out, "face",  (fx, fy - 6),         FONT, FS_SMALL, C_FACE,  TK_NORM)
    cv2.putText(out, "hair",  (hx, max(0, hy + 14)), FONT, FS_SMALL, C_HAIR, TK_NORM)

    return out


def draw_info_panel(
    frame:        np.ndarray,
    color_label:  str,
    length_label: str,
    dominant_bgr: np.ndarray,
    tips:         list[str],
    fps:          float,
) -> np.ndarray:
    """

    Shows:
    • Colour swatch + label
    • Length label
    • Up to 3 styling tips 
    • FPS counter
    """
    out = frame.copy()
    fh, fw = out.shape[:2]

    px = fw - PANEL_W - PANEL_PAD
    py = PANEL_PAD
    pw = PANEL_W
    ph = fh - 2 * PANEL_PAD

    # Semi-transparent background
    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    y = py + 26

    # Title 
    cv2.putText(out, "Hair Analysis", (px + 8, y), FONT, FS_TITLE, C_ACCENT, TK_BOLD)
    y += 6
    cv2.line(out, (px + 6, y + 4), (px + pw - 6, y + 4), (80, 80, 80), 1)
    y += 18

    # Colour swatch + label 
    swatch_c = tuple(int(c) for c in dominant_bgr)
    sw_x, sw_y, sw_s = px + 8, y, 24
    cv2.rectangle(out, (sw_x, sw_y), (sw_x + sw_s, sw_y + sw_s), swatch_c, -1)
    cv2.rectangle(out, (sw_x, sw_y), (sw_x + sw_s, sw_y + sw_s), C_TEXT,   1)
    cv2.putText(out, f"Color:  {color_label}",
                (sw_x + sw_s + 8, sw_y + 17), FONT, FS_LABEL, C_TEXT, TK_NORM)
    y += sw_s + 10
    # unicode block chars for a simple visual indicator next to the length label
    #https://mike42.me/blog/2018-06-make-better-cli-progress-bars-with-unicode-block-characters
    length_bars = {
    "short":  "█░░",
    "medium": "██░",
    "long":   "███",
}
    length_bar = length_bars.get(length_label, "?")
    cv2.putText(out, f"Length: {length_label}  {length_bar}",
                (px + 8, y), FONT, FS_LABEL, C_TEXT, TK_NORM)
    y += 6
    cv2.line(out, (px + 6, y + 4), (px + pw - 6, y + 4), (80, 80, 80), 1)
    y += 18

    
    cv2.putText(out, "Styling Tips", (px + 8, y), FONT, FS_LABEL, C_ACCENT, TK_NORM)
    y += 16

    for tip_idx, tip in enumerate(tips):
        # Bullet prefix
        prefix = f"{tip_idx + 1}. "
        # Word-wrap at ~33 characters (including prefix on first line)
        words   = tip.split()
        line    = prefix
        first   = True
        for word in words:
            limit = 33 if first else 36
            if len(line) + len(word) + 1 > limit:
                cv2.putText(out, line.rstrip(), (px + 10, y),
                            FONT, FS_SMALL, C_TEXT, TK_NORM)
                y   += 17
                line = "   " + word + " "   # indent continuation
                first = False
            else:
                line += word + " "
        if line.strip():
            cv2.putText(out, line.rstrip(), (px + 10, y),
                        FONT, FS_SMALL, C_TEXT, TK_NORM)
            y += 17
        y += 5   # gap between tips

    #FPS counter 
    fps_col = C_FPS_OK if fps >= 15 else C_FPS_BAD
    cv2.putText(out, f"FPS: {fps:.1f}",
                (px + 8, py + ph - 10), FONT, FS_LABEL, fps_col, TK_NORM)

    return out


def draw_no_face(frame: np.ndarray) -> np.ndarray:
    """Overlay a 'no face detected' message."""
    out = frame.copy()
    msg = "No face detected please centre your face in frame"
    (tw, th), _ = cv2.getTextSize(msg, FONT, FS_LABEL, TK_NORM)
    tx = max(0, (out.shape[1] - tw) // 2)
    ty = 36
    cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                  (0, 0, 0), -1)
    cv2.putText(out, msg, (tx, ty), FONT, FS_LABEL, (0, 80, 255), TK_NORM)
    return out
