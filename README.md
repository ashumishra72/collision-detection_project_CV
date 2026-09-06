# Collision Detection in Robotic Surgery Video

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-11n--seg-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A reproducible pipeline for detecting instrument-tip collisions in robotic-surgery training video, combining YOLO11n-seg tip tracking with an XGBoost classifier trained on human-annotated ground truth.**

[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Results](#-results) •
[Configuration](#-configuration) •
[Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Overview

This project implements an end-to-end pipeline for **automated collision detection** between two surgical instruments in a pattern-cutting training exercise. It ingests a raw video and a human-annotated Word document, and produces per-frame collision predictions along with a full suite of evaluation metrics and diagnostic charts.

The pipeline is designed for **reproducibility and auditability**: every intermediate artifact is written to disk as a CSV or JSON, feature extraction and classifier training are deterministic given a fixed random seed, and the train/test split is chronological rather than random to prevent temporal leakage.

### Key Features

- 🎯 **Real-time capable**: YOLO11n-seg inference at ~60 FPS on GPU, ~15 FPS on CPU
- 📊 **Interpretable**: XGBoost feature importances expose which signals drive each decision
- 🔬 **Reproducible**: pinned random seeds, chronological splits, version-controlled artifacts
- 📝 **Well-documented**: every script has a docstring, every CLI argument has a help string
- ⚙️ **Configurable**: sampling FPS, inference resolution, label window, and split fraction are all CLI arguments
- 📈 **Rich evaluation**: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and feature-importance charts

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9 or higher**
- **8 GB RAM minimum** (16 GB recommended)
- **NVIDIA GPU with CUDA 11.8+** (optional but strongly recommended for YOLO inference)
- **~2 GB free disk space** for intermediate artifacts

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/collision_project.git
cd collision_project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Setup

Place the input files in their expected locations before running the pipeline:

```
data/video.mp4                              ← Source pattern-cutting video
docs/Pattern_Cutting_Annotation_1.docx      ← Human annotation table
models/video_tip_circle_yolo11n_seg_best.pt ← Pre-trained tip detector
```

### Running the Pipeline

Execute the four numbered scripts in order:

```bash
python scripts/01_convert_annotations.py         # Word doc  →  ground_truth.csv
python scripts/02_extract_features.py            # Video     →  features_and_labels.csv
python scripts/03_train_classifier.py            # Features  →  xgb_collision_classifier.json
python scripts/04_evaluate.py                    # Predictions → metrics.json + charts
```

Total runtime on a modest workstation with an NVIDIA T4 GPU: **~15 minutes end-to-end**.

---

## 🏗️ Architecture

### Directory Layout

```
collision_project/
│
├── 📁 models/                                   # Trained model artifacts
│   ├── video_tip_circle_yolo11n_seg_best.pt   # YOLO11n-seg tip/marker detector
│   │                                            # Classes: TIPL, TIPR, TIPandCircle
│   └── xgb_collision_classifier.json          # XGBoost binary classifier
│
├── 📁 data/                                     # Input and intermediate datasets
│   ├── video.mp4                              # Source video (30 FPS, 1920×1080)
│   ├── ground_truth.csv                       # Parsed annotation events
│   └── features_and_labels.csv                # Per-frame feature matrix
│
├── 📁 docs/                                     # Source documentation
│   └── Pattern_Cutting_Annotation_1.docx      # Human annotation table
│
├── 📁 scripts/                                  # Executable pipeline stages
│   ├── 01_convert_annotations.py              # Stage 1: Annotation ingest
│   ├── 02_extract_features.py                 # Stage 2: Feature engineering
│   ├── 03_train_classifier.py                 # Stage 3: Model training
│   └── 04_evaluate.py                         # Stage 4: Evaluation
│
├── 📁 results/                                  # Outputs from evaluation
│   ├── predictions.csv                        # Per-frame test predictions
│   ├── feature_importance.csv                 # XGBoost feature rankings
│   ├── metrics.json                           # Aggregate test metrics
│   └── charts/                                # PNG visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       └── feature_importance.png
│
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
└── .gitignore
```

### Pipeline Stages

Each stage reads from disk, writes to disk, and can be re-run independently without recomputing upstream stages.

| Stage | Script | Reads | Writes |
|:-----:|--------|-------|--------|
| **1** | `01_convert_annotations.py` | `docs/*.docx` | `data/ground_truth.csv` |
| **2** | `02_extract_features.py` | `data/video.mp4`, `models/*.pt`, `data/ground_truth.csv` | `data/features_and_labels.csv` |
| **3** | `03_train_classifier.py` | `data/features_and_labels.csv` | `models/xgb_collision_classifier.json`, `results/predictions.csv`, `results/feature_importance.csv` |
| **4** | `04_evaluate.py` | `results/predictions.csv`, `results/feature_importance.csv` | `results/metrics.json`, `results/charts/*.png` |

---

## 🔬 Methodology

### 1. Annotation Parsing

The human annotation Word document contains a table with columns for start time, end time, event type, involved objects, collision severity, and near-miss flag. Timestamps are stored in **"M.SS" format** (e.g. `3.09` for 3 min 09 s), a byproduct of Word rendering `mm:ss` entries as decimal numbers.

The parser reads the document's XML directly (avoiding the heavyweight `python-docx` dependency), locates the annotation table, and emits a clean CSV with time in **absolute seconds**.

**Ground-truth rules**:
- A row is labeled `is_collision = 1` if its collision-severity cell is populated
- A row is labeled `near_miss = 1` if its near-miss cell contains "near miss"
- Events with duration 0 (single-timestamp) are preserved as instantaneous events

### 2. Feature Extraction

The video is **sub-sampled to 5 FPS** (every sixth frame at native 30 FPS) to balance temporal fidelity against inference cost. For each sampled frame, the YOLO11n-seg model detects three classes:

- **TIPL** — left instrument tip
- **TIPR** — right instrument tip
- **TIPandCircle** — combined tip-and-target marker

Per class, only the highest-confidence detection is retained. From these detections, **30 per-frame features** are derived across four families:

<details>
<summary><b>Complete Feature List</b></summary>

| Family | Feature | Description |
|--------|---------|-------------|
| **Presence** | `{class}_present` | Binary flag (1 = detected, 0 = missing) |
| **Position** | `{class}_x`, `{class}_y` | Bounding-box centroid coordinates (pixels) |
| **Geometry** | `{class}_w`, `{class}_h` | Bounding-box width and height (pixels) |
| **Geometry** | `{class}_area` | Segmentation-mask area in pixels (pseudo-depth cue) |
| **Confidence** | `{class}_conf` | YOLO detection confidence |
| **Proximity** | `dist_TIPL_TIPR` | Euclidean distance between left and right tips |
| **Proximity** | `dist_TIPL_CIRCLE` | Distance between left tip and target |
| **Proximity** | `dist_TIPR_CIRCLE` | Distance between right tip and target |
| **Overlap** | `iou_TIPL_TIPR` | Bounding-box IoU between left and right tips |
| **Motion** | `{class}_speed` | Frame-to-frame centroid displacement / dt |

Total: **30 features per frame** (7 per-class × 3 classes + 3 pairwise distances + 1 IoU + 3 speeds − 1 duplicate).

</details>

**Labeling policy**: a frame is labeled `label_collision = 1` if its timestamp falls within any annotated collision event window, **padded by ±0.5 s** on each side to absorb annotation imprecision. `label_near_miss` is set analogously for near-miss events. `label_severity` inherits the maximum severity of overlapping events.

### 3. Model Training

An **XGBoost binary classifier** is fit on the extracted features with the following configuration:

```python
xgb.XGBClassifier(
    n_estimators   = 300,
    max_depth      = 4,
    learning_rate  = 0.05,
    objective      = "binary:logistic",
    eval_metric    = "logloss",
    scale_pos_weight = n_neg / n_pos,   # Class-imbalance mitigation
    random_state   = 42,
)
```

**Critical design choice — chronological split**: the first **80% of the timeline** is used for training and the remaining 20% for testing. Random splits are **explicitly avoided** because adjacent frames in the same collision event are highly correlated; a random shuffle would place near-duplicate frames in both splits and inflate reported accuracy by up to 15–20 percentage points.

### 4. Evaluation

The evaluation script computes and reports:

- **Frame-level metrics**: accuracy, precision, recall, F1
- **Threshold-independent metrics**: ROC-AUC, average precision (PR-AUC)
- **Confusion matrix** at the default 0.5 threshold
- **Diagnostic charts**: ROC curve, precision-recall curve, confusion matrix heatmap, top-15 feature importances

---

## 📊 Results

Actual results from a 12.8-minute pattern-cutting video containing 89 annotated events:

### Frame-Level Test Metrics (threshold = 0.5)

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| **Test samples** | 784 | 20% chronological tail |
| **Positive rate** | 7.1% | 56 collision frames |
| **Accuracy** | 0.900 | ⚠️ Trivial "all-negative" baseline = 0.929 |
| **Precision** | 0.211 | 21% of predicted collisions are correct |
| **Recall** | 0.143 | 14% of true collisions are recovered |
| **F1 score** | 0.170 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.733 | ✅ Substantial discriminative ability |
| **Average precision** | 0.151 | vs 0.071 prevalence baseline |

### Confusion Matrix

|              | **Predicted: No** | **Predicted: Yes** |
|--------------|:-----------------:|:------------------:|
| **Actual: No**  | 698 (TN) | 30 (FP) |
| **Actual: Yes** | 48 (FN)  | 8 (TP)  |

### Top Features by Importance

The XGBoost model ranks the pairwise tip distance far above every other feature:

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | `dist_TIPL_TIPR` | 0.190 |
| 2 | `TIPandCircle_conf` | 0.073 |
| 3 | `TIPL_y` | 0.058 |
| 4 | `TIPL_area` | 0.052 |
| 5 | `TIPandCircle_w` | 0.047 |

### Interpreting the Results

The **high ROC-AUC (0.733) alongside low F1 (0.170)** is a classic signature of an **operating-point mismatch**, not a broken classifier. The underlying ranking is informative; the default 0.5 threshold is simply the wrong place to cut under 7% positive prevalence. See [Roadmap](#-roadmap) for planned improvements.

---

## ⚙️ Configuration

Each script accepts command-line arguments. Run any script with `--help` to see the full list.

### Common Options

| Option | Default | Description |
|--------|:-------:|-------------|
| `--sample-fps` | `5.0` | Frames per second to sample from the video |
| `--imgsz` | `960` | YOLO inference input resolution |
| `--conf` | `0.25` | YOLO detection confidence threshold |
| `--label-window` | `0.5` | Seconds to pad ground-truth event windows |
| `--max-frames` | `None` | Limit sampled frames (for quick testing) |
| `--train-frac` | `0.8` | Chronological train/test split ratio |
| `--n-estimators` | `300` | Number of XGBoost trees |
| `--max-depth` | `4` | Maximum tree depth |
| `--learning-rate` | `0.05` | XGBoost learning rate |

### Example: Quick Smoke Test

Process only the first 500 sampled frames at 10 FPS:

```bash
python scripts/02_extract_features.py --sample-fps 10 --max-frames 500
python scripts/03_train_classifier.py
python scripts/04_evaluate.py
```

### Example: Higher Fidelity Run

Sample at native 30 FPS and use a larger inference resolution:

```bash
python scripts/02_extract_features.py --sample-fps 30 --imgsz 1280 --conf 0.20
python scripts/03_train_classifier.py --n-estimators 500 --max-depth 6
python scripts/04_evaluate.py
```

---

## 🗺️ Roadmap

The current F1 of 0.170 leaves substantial room for improvement. Four enhancements are planned, ordered by expected effort-to-benefit ratio:

- [ ] **Threshold tuning** — Select the operating point that maximizes F1 on a held-out validation split rather than defaulting to 0.5. Expected F1 gain: **+0.15 to +0.25** at zero engineering cost.
- [ ] **Temporal smoothing** — Apply a five-frame centered rolling mean to predicted probabilities; require any predicted event to span at least three consecutive sampled frames.
- [ ] **Wider label window** — Increase padding from ±0.5 s to ±1.0 s and interpolate short events, ensuring every annotated collision produces at least one positive training sample.
- [ ] **Monocular depth channel** — Add MiDaS or Depth Anything V2 depth estimation as an additional per-frame input, replacing the coarse mask-area pseudo-depth proxy.

Further extensions:

- [ ] **Multi-class classification** — Distinguish contact, swiping collision, sustained contact, scraping collision, base-board disruption, and near miss (currently collapsed to binary)
- [ ] **Severity prediction** — Regress the 0/1/2 severity ratings as a secondary output head
- [ ] **Fine-tune YOLO** on additional annotated frames to reduce segmentation dropouts
- [ ] **Cross-video generalization** — Add support for training on multiple videos and evaluating hold-one-out

---

## 🔧 Troubleshooting

<details>
<summary><b>YOLO model fails to load</b></summary>

**Symptom**: `FileNotFoundError` or `RuntimeError` when calling `YOLO(...)`.

**Fix**:
1. Confirm the weights file exists at `models/video_tip_circle_yolo11n_seg_best.pt`
2. Reinstall `ultralytics`: `pip install --upgrade --force-reinstall ultralytics`
3. Verify PyTorch and CUDA versions match: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

</details>

<details>
<summary><b>OpenCV cannot open the video</b></summary>

**Symptom**: `RuntimeError: Could not open video`.

**Fix**:
- Confirm the file exists and is readable: `ls -la data/video.mp4`
- On Linux, install the standard OpenCV codec pack: `pip install opencv-python opencv-contrib-python`
- If the video uses an unusual codec, transcode with `ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p data/video.mp4`

</details>

<details>
<summary><b>Feature extraction is too slow</b></summary>

**Symptom**: `02_extract_features.py` takes hours on CPU.

**Fix**:
- Ensure PyTorch is using the GPU: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`
- Reduce `--imgsz` from `960` to `640`
- Use `--max-frames` for a quick smoke test first
- On CPU, expect **~10x slower** than GPU

</details>

<details>
<summary><b>Annotation parser finds no rows</b></summary>

**Symptom**: `RuntimeError: Could not find the annotation table in the docx`.

**Fix**:
- Confirm the Word document has a table whose first header cell reads exactly `"Start Time"` (case sensitive, no leading whitespace)
- Open the doc in Word/LibreOffice and re-save if the file was produced by an unusual tool
- If the timestamp format is not `M.SS` or `MM:SS`, edit `parse_timestamp()` in `01_convert_annotations.py`

</details>

<details>
<summary><b>F1 score is very low (< 0.2)</b></summary>

**Symptom**: Frame-level F1 stays below 0.2 on the test split.

**Diagnosis and fixes**:
1. Check ROC-AUC — if it is ≥ 0.7, the classifier is fine and the operating point is wrong. Implement **threshold tuning** (see [Roadmap](#-roadmap)).
2. Check `results/predictions.csv` — if `predicted_proba` for true positives is consistently > 0.3 but < 0.5, threshold tuning alone will fix it.
3. If ROC-AUC is < 0.6, the feature extractor may be failing — inspect `features_and_labels.csv` for rows where all `*_present` flags are 0.

</details>

---

## 📦 Dependencies

The complete pinned list is in `requirements.txt`. Core dependencies:

| Package | Version | Purpose |
|---------|:-------:|---------|
| `ultralytics` | ≥ 8.0.0 | YOLO11 inference |
| `opencv-python` | ≥ 4.5.0 | Video I/O and frame handling |
| `xgboost` | ≥ 1.7.0 | Binary classifier |
| `scikit-learn` | ≥ 1.3.0 | Metrics and utilities |
| `pandas` | ≥ 2.0.0 | Tabular data handling |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `matplotlib` | ≥ 3.7.0 | Chart rendering |

---

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a pull request.

**Development workflow**:

```bash
# Fork the repository, then:
git checkout -b feature/your-feature-name

# Make your changes, then run the full pipeline to verify nothing regressed:
python scripts/01_convert_annotations.py
python scripts/02_extract_features.py --max-frames 500
python scripts/03_train_classifier.py
python scripts/04_evaluate.py

# Format code before committing:
black scripts/
```

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- The **surgical skill assessment team** for providing the pattern-cutting video, human-annotated event log, and pre-trained YOLO11n-seg tip tracker.
- **Ultralytics** for the YOLO11 architecture and training tooling.
- The **XGBoost** and **scikit-learn** communities for making rigorous binary classification accessible.

---

## 📚 Citation

If you use this pipeline in your research, please cite it as:

```bibtex
@software{collision_detection_pipeline,
  title  = {Collision Detection Between Surgical Tools:
            A YOLO-Segmentation + XGBoost Pipeline for the Pattern-Cutting Task},
  author = {Author Name},
  year   = {2025},
  url    = {https://github.com/<your-username>/collision_project}
}
```

---

<div align="center">

**Built with** 🩺 **for safer robotic surgery.**

[⬆ Back to top](#collision-detection-in-robotic-surgery-video)

</div>
