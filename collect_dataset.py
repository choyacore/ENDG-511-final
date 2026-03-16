"""
collect_dataset.py – Custom dataset collection helper.

Captures and saves labelled hair images from the webcam.
Images are organised into:  <output_dir>/<color>/<length>/<timestamp>.jpg

This is used to build the custom team dataset that supplements
the public CelebA dataset for training/validation.

Usage
-----
    python collect_dataset.py --color brown --length medium
    python collect_dataset.py --color black --length short --out ./dataset

Controls (webcam window)
------------------------
    SPACE  – capture and save the current frame
    q      – quit and print summary

ENDG 511 – Team 14
"""

import argparse
import os
import time

import cv2

from config import CAMERA_INDEX

VALID_COLORS  = ["black", "brown", "blonde", "red", "auburn", "gray", "white"]
VALID_LENGTHS = ["short", "medium", "long"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hair Dataset Collector – ENDG 511 Team 14"
    )
    p.add_argument("--color",  choices=VALID_COLORS,  required=True,
                   help="Hair colour label for this session")
    p.add_argument("--length", choices=VALID_LENGTHS, required=True,
                   help="Hair length label for this session")
    p.add_argument("--out",    default="./dataset",
                   help="Root output directory (default: ./dataset)")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    save_dir = os.path.join(args.out, args.color, args.length)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    count   = 0
    WINDOW  = f"Dataset Collector  [{args.color} / {args.length}]"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print(f"\n[INFO] Session label  : {args.color} / {args.length}")
    print(f"[INFO] Saving to      : {save_dir}")
    print("[INFO] SPACE = capture frame |  q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # HUD overlay
        label_txt = f"Label: {args.color} / {args.length}    saved: {count}"
        cv2.rectangle(display, (0, 0), (display.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(display, label_txt, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 100), 2)
        cv2.putText(display, "SPACE = capture    q = quit",
                    (8, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            fname = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
            print(f"  [{count:04d}] Saved → {fname}")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {count} image(s) saved to {save_dir}")


if __name__ == "__main__":
    main()
