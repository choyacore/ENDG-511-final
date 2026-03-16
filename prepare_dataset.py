"""
prepare_dataset.py – Download and prepare public hair datasets.

Downloads CelebA, LFW, and a small FFHQ subset and organises them
into the folder structure expected by train_cnn.py:

    dataset/
    |-- black/
    |-- brown/
    |-- blonde/
    |-- red/
    |-- auburn/
    |-- gray/
    |-- white/
    |-- short/       (length task)
    |-- medium/
    |-- long/
    +-- unlabelled/  (for SimCLR SSL pretraining)

Sources
-------
  CelebA  : Google Drive (via gdown) -- 200k+ labelled face images
             Hair colour attributes built in to Anno/list_attr_celeba.txt
  LFW     : University of Massachusetts -- good lighting diversity
  FFHQ    : Hugging Face datasets (thumbnails 128px) -- hair texture diversity

Usage
-----
    # Download everything (recommended first run)
    python prepare_dataset.py --all

    # Individual sources
    python prepare_dataset.py --celeba
    python prepare_dataset.py --lfw
    python prepare_dataset.py --ffhq

    # Dry run -- just show what would be downloaded
    python prepare_dataset.py --all --dry-run

    # Custom output directory
    python prepare_dataset.py --all --out ./my_dataset

ENDG 511 -- Team 14
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
import tarfile
import random
from pathlib import Path

import cv2
import numpy as np

# -- Optional imports (installed if missing) ----------------------------------
def _ensure(pkg: str, import_name: str = None):
    import importlib
    name = import_name or pkg
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"[INFO] Installing {pkg}...")
        os.system(f"{sys.executable} -m pip install {pkg} -q")
        return importlib.import_module(name)

# -- Constants ----------------------------------------------------------------
DEFAULT_OUT  = "./dataset"
UNLABELLED   = "unlabelled"

# CelebA attribute column indices (0-indexed) from list_attr_celeba.txt
# Attributes present in CelebA relevant to hair:
CELEBA_ATTRS = {
    "Black_Hair":  8,
    "Blond_Hair":  9,
    "Brown_Hair": 11,
    "Gray_Hair":  17,
    # No direct auburn/red -- we derive from Black=0, Blond=0, Brown=0, Gray=0
}

# Minimum images per class before we warn
MIN_PER_CLASS = 200

# Hair colour classes matching config.py
COLOR_CLASSES  = ["black", "brown", "blonde", "red", "auburn", "gray", "white"]
LENGTH_CLASSES = ["short", "medium", "long"]


# =============================================================================
# Utility helpers
# =============================================================================

def _make_dirs(out: str, dry_run: bool = False) -> None:
    for cls in COLOR_CLASSES + LENGTH_CLASSES + [UNLABELLED]:
        d = os.path.join(out, cls)
        if not dry_run:
            os.makedirs(d, exist_ok=True)


def _save_image(src_path: str, dst_dir: str, fname: str,
                size: int = 64) -> bool:
    """Resize and save image. Returns True on success."""
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False
        img = cv2.resize(img, (size, size))
        cv2.imwrite(os.path.join(dst_dir, fname), img)
        return True
    except Exception:
        return False


def _count_class(out: str, cls: str) -> int:
    d = os.path.join(out, cls)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _print_summary(out: str) -> None:
    print("\n  Dataset summary:")
    total = 0
    for cls in COLOR_CLASSES + LENGTH_CLASSES + [UNLABELLED]:
        n = _count_class(out, cls)
        total += n
        bar = "█" * min(40, n // 10)
        print(f"    {cls:<12} {n:5d}  {bar}")
    print(f"    {'TOTAL':<12} {total:5d}\n")


# =============================================================================
# CelebA
# =============================================================================

def download_celeba(out: str, dry_run: bool = False,
                    max_per_class: int = 3000) -> None:
    """
    Download CelebA via gdown and map attribute labels to hair colour classes.

    CelebA has 40 binary attributes per image. We use:
      Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair

    Images with none of these set (or conflicting) go to unlabelled/.
    We approximate auburn as Brown_Hair=1 with high H channel (done in HSV).
    """
    print("\n" + "=" * 55)
    print("  Downloading CelebA")
    print("=" * 55)

    if dry_run:
        print("  [DRY RUN] Would download CelebA (~1.4 GB images + labels)")
        return

    gdown = _ensure("gdown")

    raw_dir = "./celeba_raw"
    os.makedirs(raw_dir, exist_ok=True)

    # -- Download attribute file ---------------------------------------------
    attr_path = os.path.join(raw_dir, "list_attr_celeba.txt")
    if not os.path.exists(attr_path):
        print("  Downloading attribute annotations...")
        # Attribute file (small, ~30 MB)
        gdown.download(
            "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
            attr_path, quiet=False
        )
    else:
        print(f"  [SKIP] Attribute file already exists: {attr_path}")

    # -- Download images archive ---------------------------------------------
    img_zip = os.path.join(raw_dir, "img_align_celeba.zip")
    img_dir = os.path.join(raw_dir, "img_align_celeba")

    if not os.path.isdir(img_dir):
        if not os.path.exists(img_zip):
            print("  Downloading CelebA images (1.4 GB -- this will take a while)...")
            gdown.download(
                "https://drive.google.com/uc?id=0B7EVK8r0v71pTUZsaXdaSnZBZzg",
                img_zip, quiet=False
            )
        print("  Extracting images...")
        with zipfile.ZipFile(img_zip, "r") as zf:
            zf.extractall(raw_dir)
    else:
        print(f"  [SKIP] Images already extracted: {img_dir}")

    # -- Parse attribute file ------------------------------------------------
    print("  Parsing attributes and copying images...")
    with open(attr_path) as f:
        lines = f.readlines()

    n_images   = int(lines[0].strip())
    attr_names = lines[1].strip().split()
    print(f"  Found {n_images} images, {len(attr_names)} attributes")

    # Map CelebA attribute name -> column index
    col = {name: i for i, name in enumerate(attr_names)}

    counts    = {cls: 0 for cls in COLOR_CLASSES + [UNLABELLED]}
    skipped   = 0

    for line in lines[2:]:
        parts  = line.strip().split()
        fname  = parts[0]
        attrs  = [int(x) for x in parts[1:]]

        black  = attrs[col["Black_Hair"]]  == 1
        blonde = attrs[col["Blond_Hair"]]  == 1
        brown  = attrs[col["Brown_Hair"]]  == 1
        gray   = attrs[col["Gray_Hair"]]   == 1

        # Assign colour class
        if black  and counts["black"]  < max_per_class:
            cls = "black"
        elif blonde and counts["blonde"] < max_per_class:
            cls = "blonde"
        elif brown  and counts["brown"]  < max_per_class:
            cls = "brown"
        elif gray   and counts["gray"]   < max_per_class:
            cls = "gray"
        elif (not black and not blonde and not brown and not gray
              and counts[UNLABELLED] < max_per_class * 2):
            cls = UNLABELLED
        else:
            skipped += 1
            continue

        src = os.path.join(img_dir, fname)
        if not os.path.exists(src):
            continue

        dst_dir = os.path.join(out, cls)
        new_fname = f"celeba_{fname}"
        if _save_image(src, dst_dir, new_fname):
            counts[cls] += 1

        if sum(counts.values()) % 1000 == 0:
            print(f"    Processed {sum(counts.values())} images...")

    # -- Approximate auburn from brown images --------------------------------
    # Auburn = warm brown; check HSV hue of sampled pixels
    print("  Detecting auburn samples from brown class...")
    brown_dir  = os.path.join(out, "brown")
    auburn_dir = os.path.join(out, "auburn")
    moved = 0

    for fname in os.listdir(brown_dir):
        if moved >= max_per_class // 3:
            break
        fpath = os.path.join(brown_dir, fname)
        img   = cv2.imread(fpath)
        if img is None:
            continue
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Auburn: hue 10-22, high saturation
        mask = cv2.inRange(hsv,
                           np.array([10, 80, 50]),
                           np.array([22, 255, 200]))
        if mask.sum() / (img.shape[0] * img.shape[1]) > 0.3:
            shutil.move(fpath, os.path.join(auburn_dir, fname))
            moved += 1

    print(f"  Moved {moved} brown -> auburn")
    print(f"  CelebA done: {counts}")


# =============================================================================
# LFW (Labeled Faces in the Wild)
# =============================================================================

def download_lfw(out: str, dry_run: bool = False) -> None:
    """
    Download LFW (Labeled Faces in the Wild) from UMass.

    LFW does not have hair colour labels, so all images go to unlabelled/
    for SimCLR SSL pretraining. This improves lighting robustness because
    LFW images span a very wide range of real-world lighting conditions.
    """
    print("\n" + "=" * 55)
    print("  Downloading LFW (Labeled Faces in the Wild)")
    print("=" * 55)

    url     = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
    raw_dir = "./lfw_raw"
    tgz     = os.path.join(raw_dir, "lfw-funneled.tgz")
    lfw_dir = os.path.join(raw_dir, "lfw_funneled")

    if dry_run:
        print(f"  [DRY RUN] Would download LFW from {url} (~180 MB)")
        return

    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.isdir(lfw_dir):
        if not os.path.exists(tgz):
            print(f"  Downloading LFW (~180 MB)...")
            import urllib.request
            urllib.request.urlretrieve(url, tgz,
                reporthook=lambda b, bs, t:
                    print(f"    {min(100, int(b*bs*100/max(t,1))):3d}%",
                          end="\r") if b % 100 == 0 else None)
            print()
        print("  Extracting LFW...")
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(raw_dir)
    else:
        print(f"  [SKIP] LFW already extracted: {lfw_dir}")

    # All LFW images -> unlabelled/ (no hair labels available)
    dst_dir = os.path.join(out, UNLABELLED)
    count   = 0
    lfw_img_dir = os.path.join(raw_dir, "lfw_funneled")
    if not os.path.isdir(lfw_img_dir):
        lfw_img_dir = lfw_dir

    for person_dir in os.listdir(lfw_img_dir):
        person_path = os.path.join(lfw_img_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(person_path, fname)
            new_fname = f"lfw_{person_dir}_{fname}"
            if _save_image(src, dst_dir, new_fname):
                count += 1

    print(f"  LFW done: {count} images -> unlabelled/")


# =============================================================================
# FFHQ (via Hugging Face -- thumbnails only)
# =============================================================================

def download_ffhq(out: str, dry_run: bool = False,
                  n_samples: int = 5000) -> None:
    """
    Download a subset of FFHQ thumbnails via Hugging Face datasets.

    FFHQ images are high-quality and very diverse in terms of hair texture
    (straight, wavy, curly, coily, locs, braids) -- directly addressing
    the hair style diversity challenge in our proposal.

    All go to unlabelled/ for SSL pretraining.
    Uses the 128px thumbnail version to keep download size manageable.
    """
    print("\n" + "=" * 55)
    print("  Downloading FFHQ thumbnails (Hugging Face)")
    print("=" * 55)

    if dry_run:
        print(f"  [DRY RUN] Would download {n_samples} FFHQ thumbnails "
              "via Hugging Face datasets")
        return

    try:
        datasets = _ensure("datasets")
    except Exception:
        print("  [WARN] Could not install 'datasets' library. "
              "Skipping FFHQ.\n"
              "  Install manually: pip install datasets")
        return

    print(f"  Loading FFHQ thumbnails ({n_samples} samples)...")
    print("  Note: First run streams from Hugging Face -- may take a few minutes.")

    try:
        ds = datasets.load_dataset(
            "Yao-Dou/FFHQ-thumbnail",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  [WARN] Could not load FFHQ dataset: {e}")
        print("  Skipping FFHQ. Try manually: "
              "https://github.com/NVlabs/ffhq-dataset")
        return

    dst_dir = os.path.join(out, UNLABELLED)
    count   = 0

    for i, sample in enumerate(ds):
        if count >= n_samples:
            break
        try:
            img_pil = sample["image"]
            fname   = f"ffhq_{i:06d}.jpg"
            fpath   = os.path.join(dst_dir, fname)
            img_np  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_np  = cv2.resize(img_np, (64, 64))
            cv2.imwrite(fpath, img_np)
            count += 1
            if count % 500 == 0:
                print(f"    Saved {count}/{n_samples}...")
        except Exception:
            continue

    print(f"  FFHQ done: {count} images -> unlabelled/")


# =============================================================================
# Validation
# =============================================================================

def validate_dataset(out: str) -> None:
    """
    Check dataset integrity and warn about under-represented classes.
    Also checks image readability and removes corrupt files.
    """
    print("\n" + "=" * 55)
    print("  Validating Dataset")
    print("=" * 55)

    corrupt  = 0
    removed  = 0

    for cls in COLOR_CLASSES + LENGTH_CLASSES + [UNLABELLED]:
        cls_dir = os.path.join(out, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(fpath)
            if img is None:
                corrupt += 1
                os.remove(fpath)
                removed += 1

    if corrupt:
        print(f"  Removed {removed} corrupt images.")
    else:
        print("  No corrupt images found.")

    # Check class balance
    print("\n  Class balance check:")
    for cls in COLOR_CLASSES:
        n = _count_class(out, cls)
        status = "OK" if n >= MIN_PER_CLASS else f"LOW (target >= {MIN_PER_CLASS})"
        print(f"    {cls:<12} {n:5d}  {status}")

    unlabelled_n = _count_class(out, UNLABELLED)
    print(f"    {UNLABELLED:<12} {unlabelled_n:5d}  "
          f"{'OK' if unlabelled_n >= 1000 else 'LOW (target >= 1000)'}")


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Hair Dataset Downloader -- ENDG 511 Team 14",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--all",     action="store_true",
                   help="Download all datasets (CelebA + LFW + FFHQ)")
    p.add_argument("--celeba",  action="store_true")
    p.add_argument("--lfw",     action="store_true")
    p.add_argument("--ffhq",    action="store_true")
    p.add_argument("--out",     default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded without doing it")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate an existing dataset")
    p.add_argument("--max-per-class", type=int, default=3000,
                   help="Max images per colour class (default: 3000)")
    p.add_argument("--ffhq-samples",  type=int, default=5000,
                   help="FFHQ thumbnails to download (default: 5000)")
    return p.parse_args()


def main():
    args = _parse_args()

    if not (args.all or args.celeba or args.lfw or args.ffhq
            or args.validate_only):
        print("Nothing to do. Use --all or --celeba / --lfw / --ffhq.")
        print("Try:  python prepare_dataset.py --all --dry-run")
        return

    print(f"\n{'='*55}")
    print(f"  Hair Dataset Preparation  |  out={args.out}")
    print(f"{'='*55}")

    if not args.validate_only:
        _make_dirs(args.out, args.dry_run)

    if args.validate_only:
        validate_dataset(args.out)
        _print_summary(args.out)
        return

    if args.all or args.celeba:
        download_celeba(args.out, args.dry_run, args.max_per_class)

    if args.all or args.lfw:
        download_lfw(args.out, args.dry_run)

    if args.all or args.ffhq:
        download_ffhq(args.out, args.dry_run, args.ffhq_samples)

    if not args.dry_run:
        validate_dataset(args.out)
        _print_summary(args.out)

    print("[DONE] Dataset ready for train_cnn.py\n")
    print("Next steps:")
    print("  1. Collect webcam images:")
    print("     python collect_dataset.py --color brown --length medium")
    print("  2. Train:")
    print("     python train_cnn.py --task color --data ./dataset")


if __name__ == "__main__":
    main()
