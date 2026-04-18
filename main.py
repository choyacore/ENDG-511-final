"""
main.py – Hair Analysis IoT System entry point.

Runs the full pipeline in a live webcam loop (or on a single image).

Usage

    # Live webcam (default)
    python main.py

    # Specific camera index
    python main.py --camera 1

    # Single image (for debugging)
    python main.py --image path/to/photo.jpg

Keyboard shortcuts (webcam mode)

    q / ESC  – quit
    s        – save annotated frame as PNG
    p        – pause / resume

#ENDG 410 webcam code inspired
"""

import argparse
import sys
import time

import cv2
import numpy as np

from config          import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from hair_detector   import FaceHairDetector
from color_classifier import classify_color
from length_classifier import classify_length
from recommender     import get_tips
from utils           import draw_roi_boxes, draw_info_panel, draw_no_face


# ─── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hair Analysis IoT System – ENDG 511 Team 14"
    )
    p.add_argument("--camera", type=int,  default=CAMERA_INDEX,
                   help="Webcam device index (default: 0)")
    p.add_argument("--width",  type=int,  default=FRAME_WIDTH,
                   help="Capture width in pixels (default: 640)")
    p.add_argument("--height", type=int,  default=FRAME_HEIGHT,
                   help="Capture height in pixels (default: 480)")
    p.add_argument("--image",  type=str,  default=None,
                   help="Run on a single image file instead of webcam")
    p.add_argument("--interval", type=int, default=2,
                   help="Run detection every N frames (higher = faster, default: 2)")
    return p.parse_args()


# ─── Single-frame pipeline ────────────────────────────────────────────────────

def run_pipeline(frame_bgr: np.ndarray, detector: FaceHairDetector):
    """
    Runs the complete hair analysis pipeline on one frame.

    Returns(color_label, length_label, dominant_bgr, tips, roi)
    or None if no face was detected.
    """
    roi = detector.detect(frame_bgr)
    if roi is None:
        return None

    color_label, dominant_bgr = classify_color(frame_bgr, roi.hair_box)
    length_label, ratio       = classify_length(frame_bgr, roi, dominant_bgr)
    tips                      = get_tips(color_label, length_label)

    return color_label, length_label, dominant_bgr, tips, roi, ratio


# ─── Webcam loop ──────────────────────────────────────────────────────────────

def run_webcam(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}.")
        sys.exit(1)

    print(f"[INFO] Camera {args.camera} opened  "
          f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    print("[INFO] Press  q / ESC  to quit |  s  to save frame |  p  to pause")

    WINDOW = "Hair Analysis – ENDG 511 Team 14"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Cached pipeline result (re-run every `args.interval` frames)
    cache        = None          # last valid pipeline result
    frame_count  = 0
    fps          = 0.0
    t_prev       = time.perf_counter()
    paused       = False

    with FaceHairDetector() as detector:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Frame capture failed – retrying …")
                    continue

                frame_count += 1

                # Run detection every N frames to maintain high FPS
                if frame_count % args.interval == 0:
                    result = run_pipeline(frame, detector)
                    if result is not None:
                        cache = result   # update cache on success

                # FPS (exponential moving average)
                t_now  = time.perf_counter()
                dt     = max(t_now - t_prev, 1e-9)
                fps    = 0.85 * fps + 0.15 * (1.0 / dt)
                t_prev = t_now

                # Render
                if cache is not None:
                    color_label, length_label, dominant_bgr, tips, roi, ratio = cache
                    display = draw_roi_boxes(frame, roi)
                    display = draw_info_panel(
                        display, color_label, length_label,
                        dominant_bgr, tips, fps
                    )
                else:
                    display = draw_no_face(frame)
                    cv2.putText(display, f"FPS: {fps:.1f}",
                                (8, display.shape[0] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 200, 0), 1)

                cv2.imshow(WINDOW, display)
            # end if not paused

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):       # q or ESC → quit
                break
            elif key == ord('s'):            # s → save frame
                if not paused:
                    fname = f"hair_capture_{int(time.time())}.png"
                    cv2.imwrite(fname, display)
                    print(f"[INFO] Saved → {fname}")
            elif key == ord('p'):            # p → pause/resume
                paused = not paused
                status = "PAUSED" if paused else "RUNNING"
                print(f"[INFO] {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


# ─── Single-image mode ────────────────────────────────────────────────────────

def run_image(image_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Analysing image: {image_path}")

    with FaceHairDetector() as detector:
        result = run_pipeline(frame, detector)

    if result is None:
        print("[INFO] No face detected in image.")
        cv2.imshow("Result", draw_no_face(frame))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    color_label, length_label, dominant_bgr, tips, roi, ratio = result

    # Console output
    print("\n" + "─" * 46)
    print(f"  Hair Color  : {color_label}")
    print(f"  Hair Length : {length_label}  (ratio = {ratio:.2f})")
    print(f"  Styling Tips:")
    for i, tip in enumerate(tips, 1):
        print(f"    {i}. {tip}")
    print("─" * 46)

    display = draw_roi_boxes(frame, roi)
    display = draw_info_panel(
        display, color_label, length_label,
        dominant_bgr, tips, fps=0.0
    )

    cv2.imshow("Hair Analysis – ENDG 511 Team 14", display)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_webcam(args)
