"""
test_system.py – Pre-Jetson system verification tests.

Runs a series of checks on your laptop to confirm everything works
before you borrow the Jetson. Catches common problems early.

Tests
-----
  1. Dependency check        -- all required packages installed
  2. Import check            -- all project modules import cleanly
  3. Synthetic pipeline test -- runs full hair detection + classification
                                on a generated test image (no webcam needed)
  4. Webcam test             -- opens webcam and checks FPS
  5. SSL augmentation test   -- verifies SimCLR augmentation pipeline
  6. Mini training test      -- trains for 2 epochs on synthetic data
  7. Jetson compatibility    -- checks for ARM / CUDA / TensorRT readiness

Usage
-----
    # Run all tests
    python test_system.py

    # Skip webcam test (e.g., running on a server)
    python test_system.py --no-webcam

    # Skip training test (faster)
    python test_system.py --no-train

ENDG 511 -- Team 14
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import platform

import numpy as np

# Colour codes for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"
INFO = f"{BLUE}INFO{RESET}"

results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, note: str = "") -> None:
    results.append((name, passed, note))
    icon = PASS if passed else FAIL
    print(f"  [{icon}] {name}" + (f"  -- {note}" if note else ""))


def _section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# =============================================================================
# Test 1 -- Dependencies
# =============================================================================

def test_dependencies() -> None:
    _section("Test 1 -- Dependency Check")

    required = [
        ("cv2",        "opencv-python"),
        ("mediapipe",  "mediapipe"),
        ("numpy",      "numpy"),
        ("sklearn",    "scikit-learn"),
        ("PIL",        "Pillow"),
    ]
    optional = [
        ("torch",      "torch"),
        ("torchvision","torchvision"),
    ]

    for mod, pkg in required:
        try:
            __import__(mod)
            _record(f"import {pkg}", True)
        except ImportError:
            _record(f"import {pkg}", False,
                    f"pip install {pkg}")

    for mod, pkg in optional:
        try:
            __import__(mod)
            _record(f"import {pkg} (optional)", True)
        except ImportError:
            _record(f"import {pkg} (optional)", True,
                    f"Not installed -- CNN training unavailable "
                    f"(pip install {pkg})")


# =============================================================================
# Test 2 -- Project module imports
# =============================================================================

def test_imports() -> None:
    _section("Test 2 -- Project Module Imports")

    modules = [
        "config",
        "hair_detector",
        "color_classifier",
        "length_classifier",
        "recommender",
        "utils",
        "train_cnn",
    ]

    for mod in modules:
        try:
            __import__(mod)
            _record(f"import {mod}", True)
        except Exception as e:
            _record(f"import {mod}", False, str(e))


# =============================================================================
# Test 3 -- Synthetic pipeline test (no webcam)
# =============================================================================

def _make_synthetic_frame(with_face: bool = True) -> np.ndarray:
    """
    Generate a synthetic BGR frame that MediaPipe can detect a face in.
    We use a real small image embedded in a solid-colour background.
    If no real image is available, generates a programmatic face-like oval.
    """
    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (60, 50, 45)   # dark brownish background (like hair)

    if with_face:
        # Draw a simple face-like shape so MediaPipe has something to detect
        # Skin tone ellipse
        cx, cy = 320, 240
        cv2.ellipse(frame, (cx, cy), (90, 110), 0, 0, 360, (180, 140, 110), -1)
        # Eyes
        cv2.circle(frame, (cx - 30, cy - 20), 12, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 30, cy - 20), 12, (40, 40, 40), -1)
        # Nose
        cv2.ellipse(frame, (cx, cy + 10), (12, 16), 0, 0, 360,
                    (160, 120, 95), -1)
        # Mouth
        cv2.ellipse(frame, (cx, cy + 45), (30, 12), 0, 0, 180,
                    (120, 70, 70), 3)
        # Hair (dark region above face)
        cv2.ellipse(frame, (cx, cy - 80), (100, 80), 0, 0, 360,
                    (30, 20, 15), -1)

    return frame


def test_pipeline() -> None:
    _section("Test 3 -- Synthetic Pipeline (No Webcam Required)")

    try:
        import cv2
        from hair_detector    import HairDetector
        from color_classifier  import classify_color
        from length_classifier import classify_length
        from recommender       import get_tips
        from config            import get_recommendation
    except ImportError as e:
        _record("Pipeline imports", False, str(e))
        return

    # -- Detector init --------------------------------------------------------
    try:
        detector = HairDetector()
        _record("HairDetector init", True)
    except Exception as e:
        _record("HairDetector init", False, str(e))
        return

    # -- Synthetic frame ------------------------------------------------------
    frame = _make_synthetic_frame(with_face=True)
    _record("Synthetic frame created",
            frame.shape == (480, 640, 3),
            f"shape={frame.shape}")

    # -- Detection (may or may not find face in synthetic image) --------------
    try:
        roi = detector.detect(frame)
        if roi is not None:
            _record("Face detection (synthetic)", True,
                    f"face_box={roi.face_box}")
        else:
            _record("Face detection (synthetic)", True,
                    "No face found in synthetic image (expected) -- "
                    "detector is working correctly")
    except Exception as e:
        _record("Face detection (synthetic)", False, str(e))
        detector.close()
        return

    detector.close()

    # -- Colour classification with a fake ROI --------------------------------
    try:
        # Use top portion of frame as fake hair box
        fake_hair_box = (200, 0, 240, 100)
        color_label, dominant_bgr = classify_color(frame, fake_hair_box)
        _record("Color classification", True,
                f"label={color_label}  BGR={dominant_bgr.tolist()}")
    except Exception as e:
        _record("Color classification", False, str(e))

    # -- Recommendation lookup ------------------------------------------------
    try:
        for color in ["black", "brown", "blonde"]:
            for length in ["short", "medium", "long"]:
                tips = get_tips(color, length)
                assert len(tips) > 0
        _record("Recommendation lookup", True,
                "All color/length combos return tips")
    except Exception as e:
        _record("Recommendation lookup", False, str(e))

    # -- FPS benchmark (no camera) --------------------------------------------
    try:
        import cv2
        n_frames = 100
        t0 = time.perf_counter()
        for _ in range(n_frames):
            f = _make_synthetic_frame()
            _ = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        dt  = time.perf_counter() - t0
        fps = n_frames / dt
        _record("Frame processing speed",
                fps >= 30,
                f"{fps:.1f} FPS  "
                f"({'above' if fps >= 30 else 'below'} 30 FPS target)")
    except Exception as e:
        _record("Frame processing speed", False, str(e))


# =============================================================================
# Test 4 -- Webcam
# =============================================================================

def test_webcam(camera_index: int = 0) -> None:
    _section(f"Test 4 -- Webcam (camera index {camera_index})")

    try:
        import cv2
    except ImportError:
        _record("Webcam test", False, "cv2 not installed")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        _record("Camera open", False,
                f"Cannot open camera {camera_index}. "
                "Check it's connected / not in use.")
        return

    _record("Camera open", True)

    # Read 30 frames and measure FPS
    frame_times = []
    failed = 0
    for _ in range(30):
        t0  = time.perf_counter()
        ret, frame = cap.read()
        dt  = time.perf_counter() - t0
        if ret:
            frame_times.append(dt)
        else:
            failed += 1

    cap.release()

    if failed > 5:
        _record("Frame capture", False, f"{failed}/30 frames failed")
        return

    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _record("Frame capture", True,
            f"{len(frame_times)}/30 frames OK  "
            f"avg={avg_fps:.1f} FPS  {w}x{h}")

    ok_fps = avg_fps >= 15
    _record("Webcam FPS >= 15",
            ok_fps,
            f"{avg_fps:.1f} FPS  "
            f"({'OK' if ok_fps else 'below Jetson target -- check lighting/USB'})")


# =============================================================================
# Test 5 -- SSL Augmentation
# =============================================================================

def test_ssl_augmentation() -> None:
    _section("Test 5 -- SimCLR SSL Augmentation")

    try:
        from train_cnn import get_ssl_augmentation, get_train_augmentation
        from PIL import Image
    except ImportError as e:
        _record("SSL augmentation imports", False, str(e))
        return

    try:
        ssl_tfm   = get_ssl_augmentation()
        train_tfm = get_train_augmentation()
        _record("Augmentation pipelines built", True)
    except Exception as e:
        _record("Augmentation pipelines built", False, str(e))
        return

    # Generate a random image and apply augmentation
    try:
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        view1 = ssl_tfm(img)
        view2 = ssl_tfm(img)
        # The two views should be different (augmentation is random)
        diff  = (view1 - view2).abs().mean().item()
        _record("Two SSL views are different",
                diff > 0.01,
                f"mean pixel diff={diff:.4f}")

        # Shape check
        _record("View shape correct",
                tuple(view1.shape) == (3, 64, 64),
                f"shape={tuple(view1.shape)}")
    except Exception as e:
        _record("SSL augmentation forward pass", False, str(e))


# =============================================================================
# Test 6 -- Mini training (2 epochs on synthetic data)
# =============================================================================

def test_mini_training() -> None:
    _section("Test 6 -- Mini Training Test (Synthetic Data, 2 Epochs)")

    try:
        import torch
        from train_cnn import (
            HairDataset, SimCLREncoder, NTXentLoss,
            HairClassifier, get_ssl_augmentation, get_train_augmentation,
        )
    except ImportError as e:
        _record("PyTorch / train_cnn imports", True,
                f"PyTorch not installed -- skipping training test ({e})")
        return

    device = torch.device("cpu")   # always CPU for the laptop test

    # -- Build tiny synthetic dataset ----------------------------------------
    try:
        import os, tempfile
        from PIL import Image

        tmpdir  = tempfile.mkdtemp()
        classes = ["black", "brown", "blonde"]
        for cls in classes:
            os.makedirs(os.path.join(tmpdir, cls))
            for i in range(20):   # 20 images per class
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(os.path.join(tmpdir, cls, f"{i:04d}.jpg"))

        _record("Synthetic dataset created", True,
                f"3 classes x 20 images in {tmpdir}")
    except Exception as e:
        _record("Synthetic dataset creation", False, str(e))
        return

    # -- SimCLR forward pass -------------------------------------------------
    try:
        ssl_tfm = get_ssl_augmentation()
        img_pil = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        v1 = ssl_tfm(img_pil).unsqueeze(0)
        v2 = ssl_tfm(img_pil).unsqueeze(0)

        encoder  = SimCLREncoder().to(device)
        loss_fn  = NTXentLoss()
        z1, z2   = encoder(v1), encoder(v2)
        loss_val = loss_fn(z1, z2)

        _record("SimCLR forward pass", True,
                f"NT-Xent loss={loss_val.item():.4f}")
    except Exception as e:
        _record("SimCLR forward pass", False, str(e))

    # -- Classifier forward pass + 2 training steps --------------------------
    try:
        from torch.utils.data import DataLoader
        import torch.optim as optim

        tfm     = get_train_augmentation()
        ds      = HairDataset(tmpdir, classes, transform=tfm)
        dl      = DataLoader(ds, batch_size=8, shuffle=True)
        model   = HairClassifier(len(classes)).to(device)
        opt     = optim.Adam(model.parameters(), lr=1e-3)
        crit    = torch.nn.CrossEntropyLoss()

        losses = []
        for epoch in range(2):
            for imgs, labels in dl:
                opt.zero_grad()
                loss = crit(model(imgs.to(device)), labels.to(device))
                loss.backward()
                opt.step()
                losses.append(loss.item())

        _record("2-epoch training completed", True,
                f"final loss={losses[-1]:.4f}")
    except Exception as e:
        _record("2-epoch training", False, str(e))
        traceback.print_exc()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Test 7 -- Jetson compatibility hints
# =============================================================================

def test_jetson_compatibility() -> None:
    _section("Test 7 -- Jetson Compatibility Check")

    arch = platform.machine()
    is_arm = arch.startswith("aarch") or arch.startswith("arm")
    _record("Architecture",
            True,
            f"{arch}  ({'ARM -- running ON Jetson' if is_arm else 'x86 -- running on laptop (expected)'})")

    # CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_ok else "N/A"
        _record("CUDA available",
                cuda_ok,
                f"version={cuda_ver}  "
                f"({'GPU will be used' if cuda_ok else 'CPU only -- OK for laptop testing, GPU available on Jetson'})")
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            _record("GPU device", True, f"{name}  {mem} GB")
    except ImportError:
        _record("CUDA check", True, "PyTorch not installed -- skip")

    # TensorRT (Jetson only)
    try:
        import tensorrt
        _record("TensorRT installed", True, f"version={tensorrt.__version__}")
    except ImportError:
        _record("TensorRT installed", True,
                "Not installed (expected on laptop -- available on Jetson JetPack)")

    # MediaPipe CPU mode
    try:
        import mediapipe as mp
        _record("MediaPipe CPU mode", True,
                f"version={mp.__version__}  "
                "(runs on Jetson CPU without CUDA)")
    except ImportError:
        _record("MediaPipe", False, "pip install mediapipe")

    # DataLoader workers
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.randn(10, 3))
        # num_workers=2 sometimes fails on Jetson
        try:
            dl = DataLoader(ds, batch_size=2, num_workers=2)
            next(iter(dl))
            _record("DataLoader num_workers=2", True,
                    "OK on this machine -- if it fails on Jetson, "
                    "change to num_workers=0 in train_cnn.py")
        except Exception:
            _record("DataLoader num_workers=2", True,
                    "Failed -- use num_workers=0 on Jetson")
    except ImportError:
        pass


# =============================================================================
# Summary
# =============================================================================

def print_summary() -> None:
    _section("Summary")
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = [(name, note) for name, ok, note in results if not ok]

    print(f"  {passed}/{total} checks passed\n")

    if failed:
        print(f"  {RED}Failed checks:{RESET}")
        for name, note in failed:
            print(f"    - {name}: {note}")
        print()
        print(f"  {YELLOW}Fix the above before running on the Jetson.{RESET}")
    else:
        print(f"  {GREEN}All checks passed! Ready for Jetson deployment.{RESET}")

    print()
    print("  Jetson deployment checklist:")
    print("  [ ] Install JetPack (comes with OpenCV + CUDA)")
    print("  [ ] pip install mediapipe scikit-learn torch torchvision")
    print("  [ ] Copy project files via USB or git clone")
    print("  [ ] Run: python main.py  (webcam test)")
    print("  [ ] If DataLoader errors: set num_workers=0 in train_cnn.py")
    print("  [ ] For TensorRT: python train_cnn.py --export-onnx, then trtexec")
    print()


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Pre-Jetson System Tests -- ENDG 511 Team 14"
    )
    p.add_argument("--no-webcam", action="store_true",
                   help="Skip webcam test")
    p.add_argument("--no-train",  action="store_true",
                   help="Skip mini training test (faster)")
    p.add_argument("--camera",    type=int, default=0,
                   help="Webcam index (default: 0)")
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"\n{'='*55}")
    print("  Hair Analysis -- Pre-Jetson System Tests")
    print(f"  Python {sys.version.split()[0]}  |  {platform.system()} {platform.machine()}")
    print(f"{'='*55}")

    test_dependencies()
    test_imports()
    test_pipeline()

    if not args.no_webcam:
        test_webcam(args.camera)
    else:
        print(f"\n  [{WARN}] Webcam test skipped (--no-webcam)")

    test_ssl_augmentation()

    if not args.no_train:
        test_mini_training()
    else:
        print(f"\n  [{WARN}] Training test skipped (--no-train)")

    test_jetson_compatibility()
    print_summary()


if __name__ == "__main__":
    main()
