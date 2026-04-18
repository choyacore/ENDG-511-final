# Hair Analysis IoT System
**ENDG 511 – Industrial IoT Systems & Artificial Intelligence | Team 14**

Darren Taylor · Naishah Adetunji · Sehba Samman

---

## Overview

An edge-based IoT vision system that detects a person's **hair colour** and **hair length** from a webcam feed and provides real-time styling recommendations.

```
Camera → Face Detection → Hair ROI → Colour Classification ─┐
                                   └→ Length Classification ─┴→ Recommendations → Display
```

---

## Project Structure

```
hair_analysis/
├── main.py              # Entry point – webcam loop or single-image mode
├── config.py            # All constants, thresholds, and recommendation rules
├── hair_detector.py     # MediaPipe FaceMesh – face + hair ROI extraction
├── color_classifier.py  # KMeans + HSV/LAB – dominant colour classification (Sehba)
├── length_classifier.py # Geometric + LAB scan – length estimation (Naishah)
├── recommender.py       # Rule-based styling tips engine
├── utils.py             # OpenCV drawing helpers (bounding boxes, info panel)
├── collect_dataset.py   # Custom dataset collection via webcam
├── train_cnn.py         # MobileNetV2 CNN trainer + ONNX export for Jetson
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Jetson Orin Nano**: Install OpenCV with CUDA support from JetPack, then install
> mediapipe and scikit-learn via pip.

### 2. Run the live system

```bash
python main.py                    # default webcam (index 0)
python main.py --camera 1         # alternate camera
python main.py --image photo.jpg  # single-image debug mode
```

### 3. Keyboard controls (webcam mode)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save annotated frame as PNG |
| `p` | Pause / resume |

---

## Pipeline Details

### Hair Detection (`hair_detector.py`)
- Uses **MediaPipe FaceMesh** (468 landmarks, real-time).
- Derives the forehead boundary and chin position from specific landmark indices.
- Produces three bounding boxes:
  - **Face box** — for display
  - **Hair colour box** — region above the forehead (used by colour classifier)
  - **Full-head strip** — full-height strip at face x-extent (used by length classifier)

### Colour Classification (`color_classifier.py`)  *(Sehba Samman)*
- Crops the hair ROI and masks out skin-tone pixels (reduces forehead bleed-in).
- Finds the **dominant colour** using KMeans clustering (k = 3, subsampled to 2 500 px).
- Converts the dominant pixel to **HSV** and matches against lookup ranges in `config.py`.
- Supported labels: `black`, `brown`, `blonde`, `red`, `auburn`, `gray`, `white`, `dark`.

### Length Classification (`length_classifier.py`)  *(Naishah Adetunji)*
- Scans downward from the chin to find the lowest row containing hair-coloured pixels
  (LAB colour-distance threshold = 32).
- Computes: `ratio = (lowest_hair_y − forehead_y) / (chin_y − forehead_y)`
- Thresholds (configurable in `config.py`):
  - `ratio < 1.20` → **short**
  - `ratio < 1.70` → **medium**
  - otherwise → **long**

### Recommendation Engine (`recommender.py`)
- Pure rule-based lookup: `(color, length) → list[str]`.
- All rules defined in `config.py` — easy to extend.

---

## Custom Dataset Collection

Run this for each colour/length combination you want to collect:

```bash
python collect_dataset.py --color brown --length medium --out ./dataset
# SPACE = capture | q = quit
```

Saves to: `./dataset/<color>/<length>/<timestamp>.jpg`

---

## CNN Training (Advanced Goal)

Once you have collected enough data (≥ 50 images per class recommended):

```bash
# Train colour classifier
python train_cnn.py --task color --data ./dataset --epochs 20

# Train length classifier
python train_cnn.py --task length --data ./dataset --epochs 20

# Export ONNX for Jetson TensorRT
python train_cnn.py --task color --export-onnx
```

Model is saved as `hair_<task>_mobilenetv2.pt`.
ONNX export is saved as `hair_<task>_mobilenetv2.onnx`.

To convert for Jetson inference:
```bash
trtexec --onnx=hair_color_mobilenetv2.onnx --saveEngine=hair_color.engine --fp16
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Classification accuracy | ≥ 80 % | Evaluated on labelled CelebA subset |
| FPS (desktop) | ≥ 30 | Displayed live in green |
| FPS (Jetson Orin Nano) | ≥ 15 | Minimum per spec; displayed in red if below |
| Inference latency | < 67 ms | At 15 FPS budget |

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Face landmarks | MediaPipe FaceMesh | Runs on Jetson, no GPU required for CPU mode |
| Colour clustering | KMeans k=3 | Extracts dominant hair colour robustly |
| Colour space | HSV primary, LAB for length scan | HSV intuitive for hue matching; LAB perceptually uniform for distance |
| CNN architecture | MobileNetV2 | Small (14 MB), fast on Jetson, high accuracy |
| Detection cadence | Every 2nd frame | Doubles effective FPS without visible lag |

---

## References

1. CelebA dataset: https://github.com/ZhangYuanhan-AI/CelebA-Spoof  
2. Borza et al., "Deep Learning for Hair Segmentation," LNCS 2018  
3. Benchmarking DL on Jetson Nano: https://arxiv.org/html/2406.17749v1  
4. IBM – Model size vs accuracy: https://www.ibm.com/think/insights/are-bigger-language-models-better  
5. Han et al., "Deep Compression," arXiv:1510.00149  
6. Bora et al., "LAB vs HSV for segmentation," arXiv:1506.01472  
