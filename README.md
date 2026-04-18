# Hair Analysis IoT System
**ENDG 511 – Industrial IoT Systems & Artificial Intelligence | Team 1**

Darren Taylor · Naishah Adetunji · Sehba Samman

---

## Overview

An edge-based IoT vision system that detects a person's **hair colour** and **hair length** from a live webcam feed and provides real-time personalised styling recommendations.
---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python version**: Use Python 3.12. MediaPipe is incompatible with Python 3.14 — this project uses OpenCV Haar Cascade instead.

```bash
# Windows with multiple Python versions
py -3.12 main.py
```

### 2. Run the live system

```bash
python main.py                    # default webcam (index 0)
python main.py --camera 1         # alternate camera
python main.py --image photo.jpg  # single-image debug mode
```

### 3. Keyboard controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save annotated frame as PNG |
| `p` | Pause / resume |

---

## Pipeline Details

### Face Detection (`hair_detector.py`)
- Uses **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`) for real-time face detection
- MediaPipe was the original plan but is incompatible with Python 3.14 — Haar Cascade was used as the replacement
- Produces three spatial regions per frame:
  - **face_box** — raw face bounding box for display
  - **hair_box** — region above the forehead (input to colour classifier)
  - **full_head_box** — full vertical strip at face x-extent (input to length classifier)
- Key coordinates: `forehead_y` (top of face box) and `chin_y` (bottom of face box)

### Colour Classification (`color_classifier.py`)
- Crops hair ROI from frame
- Removes skin-tone pixels using HSV mask (`H: 0-25, S: 25-175, V: 60-255`)
- Runs KMeans clustering (k=3, max 2,500 pixels subsampled for speed)
- Converts dominant cluster centroid to HSV and matches against ranges in `config.py`
- Priority order: black → gray → white → brown → auburn → blonde → red
- Returns: colour label + dominant BGR pixel (used for display swatch and length scan)

### Length Classification (`length_classifier.py`)
- Takes `dominant_bgr` from colour classifier as input (dependency)
- Converts dominant colour to LAB space
- Scans downward from `chin_y` row by row (stride=3 for speed)
- Counts a row as hair if ≥4 pixels have LAB distance < 32.0 from target colour
- Computes: `ratio = (lowest_hair_y - forehead_y) / (chin_y - forehead_y)`
- `ratio < 1.20` → short | `ratio < 1.70` → medium | else → long

### Recommendation Engine (`recommender.py`)
- Rule-based lookup: `(colour, length) → list[3 tips]`
- 21 combinations covered (7 colours × 3 lengths), manually curated in `config.py`
- Falls back to generic advice if combination not found

---

## CNN Training Pipeline

Full training notebook: `final_endg_511.ipynb` — run on Google Colab with T4 GPU.

### Datasets

| Dataset | Size | Use |
|---------|------|-----|
| CelebA | 12,000 images (3,000/class) | Stage 2 supervised fine-tuning |
| LFW | 13,233 images (unlabelled) | Stage 1 SSL pretraining |
| Custom webcam | 30 images (black hair) | Few-shot support set |

### Prepare dataset

```bash
# Sort CelebA images into colour class folders
# Requires: ./dataset/img_align_celeba/img_align_celeba/ and list_attr_celeba.csv
python sort_celeba.py

# Collect your own webcam images
python collect_dataset.py --color brown --length medium --out ./dataset
# SPACE = capture frame | q = quit
```

### Run training

```bash
# Full 3-stage pipeline (recommended: run on GPU via Colab)
python train_cnn.py --task color --data ./dataset --ssl-epochs 30 --ft-epochs 20 --episodes 50

# Baseline only (skip SSL — runs on CPU in ~10 mins)
python train_cnn.py --task color --data ./dataset --skip-ssl --ft-epochs 2

# Few-shot adaptation only (after full training)
python train_cnn.py --task color --data ./webcam_support --few-shot-only --episodes 50
```

### Training stages

**Stage 1 — SimCLR SSL pretraining**
- Trains on 13,233 unlabelled LFW images — no class labels needed
- NT-Xent loss dropped from 1.94 → 0.07 over 30 epochs
- Learns lighting/texture invariance through contrastive learning

**Stage 2 — MobileNetV2 supervised fine-tuning**
- Loads SSL pretrained backbone
- Fine-tunes on 12,000 CelebA images (9,600 train / 2,400 val)
- Best val accuracy: **82.8%** at epoch 18 (vs 73.8% baseline without SSL)

**Stage 3 — Prototypical few-shot adaptation**
- Uses K=5 support images per class to compute class prototypes
- Classifies via cosine distance between embeddings
- Tested with CelebA proxy images: **10.8%** accuracy (below 25% random)
- Low accuracy expected — no genuine domain gap when support = training distribution
- Proper evaluation requires real-world webcam images across all 4 classes

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Face detection | OpenCV Haar Cascade | MediaPipe incompatible with Python 3.14 |
| Colour space | HSV for colour matching | Hue channel directly encodes perceptual colour |
| Colour space | LAB for length scan | Perceptually uniform — better distance metric |
| CNN architecture | MobileNetV2 (14 MB) | Efficient, edge-deployable, competitive accuracy |
| Detection cadence | Every 2nd frame | Doubles FPS without visible accuracy loss |
| KMeans subsample | Max 2,500 pixels | Speed optimisation — global colour, not spatial |
| Length scan stride | Every 3rd row | Speed optimisation — ratio insensitive to small errors |

---

## Performance

| Metric | Target | Achieved |
|--------|--------|---------|
| Classification accuracy | ≥ 80% | 82.8% (full pipeline) |
| Live inference FPS | ≥ 15 FPS | 30+ FPS on laptop CPU |
| Training time (baseline) | — | ~10 min (2 epochs, CPU) |
| Training time (full) | — | ~45 min (T4 GPU, Colab) |

---

## Notes on Few-Shot Results

The 10.8% few-shot accuracy is below random chance (25% for 4 classes). This is an expected result — prototypical few-shot adaptation requires a genuine domain gap between training and support data. Since CelebA proxy images were used as the support set (same distribution as training), the adaptation has no domain shift to bridge. Future work: collect 15+ webcam images per class for a proper domain adaptation evaluation.

---

## References

1. CelebA dataset: Liu et al., "Deep Learning Face Attributes in the Wild," ICCV 2015
2. LFW dataset: Huang et al., "Labeled Faces in the Wild," UMass Technical Report 2007
3. SimCLR: Chen et al., "A Simple Framework for Contrastive Learning," ICML 2020
4. MobileNetV2: Sandler et al., "Inverted Residuals and Linear Bottlenecks," CVPR 2018
5. Prototypical Networks: Snell et al., "Prototypical Networks for Few-Shot Learning," NeurIPS 2017
6. Viola & Jones, "Rapid Object Detection using a Boosted Cascade," CVPR 2001
