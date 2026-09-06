# Collision Detection Project

A computer vision pipeline for detecting collisions in a pattern-cutting video using YOLO segmentation and XGBoost classification.

## Pipeline

```text
Video
  ↓
YOLO Segmentation
  ↓
Feature Extraction
  ↓
XGBoost Classifier
  ↓
Collision Prediction
  ↓
Evaluation
```

## Project Structure

```text
collision_project/
├── models/
│   ├── video_tip_circle_yolo11n_seg_best.pt
│   └── xgb_collision_classifier.json
│
├── data/
│   ├── video.mp4
│   ├── ground_truth.csv
│   └── features_and_labels.csv
│
├── docs/
│   └── Pattern_Cutting_Annotation_1.docx
│
├── scripts/
│   ├── 01_convert_annotations.py
│   ├── 02_extract_features.py
│   ├── 03_train_classifier.py
│   ├── 04_evaluate.py
│   └── 05_visualize_detections.py
│
└── results/
    ├── predictions.csv
    ├── feature_importance.csv
    ├── metrics.json
    └── charts/
```

## Environment Setup

Open PowerShell and go to the project folder:

```powershell
cd C:\Users\ashu1\OneDrive\Desktop\collision_project
```

Create the Python virtual environment:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Run the Complete Pipeline

Run the scripts in the following order:

### 1. Convert Annotations

```powershell
python scripts\01_convert_annotations.py
```

Converts the annotation document into:

```text
data/ground_truth.csv
```

### 2. Extract Features

```powershell
python scripts\02_extract_features.py --sample-fps 5
```

Extracts YOLO-based features and creates:

```text
data/features_and_labels.csv
```

### 3. Train the Classifier

```powershell
python scripts\03_train_classifier.py
```

Trains the XGBoost collision classifier and saves the model and predictions.

### 4. Evaluate the Model

```powershell
python scripts\04_evaluate.py
```

Creates evaluation metrics and charts in:

```text
results/
```

## View YOLO Detections

To open the source video and view the YOLO detections:

```powershell
python scripts\05_visualize_detections.py
```

To start at a specific timestamp, for example 48 seconds:

```powershell
python scripts\05_visualize_detections.py --start-sec 48
```

### Video Controls

```text
Space → Pause / Resume
N     → Next frame
Q     → Quit
```

## Ground Truth

The source annotation file is:

```text
docs/Pattern_Cutting_Annotation_1.docx
```

The timestamps use the `M.SS` format.

Example:

```text
3.09 = 3 minutes 9 seconds
```

The annotation conversion script creates:

```text
data/ground_truth.csv
```

## Features

The system extracts visual features from the YOLO detections, including:

- TIPL/TIPR positions
- Distance between tips
- Distance change over time
- Bounding-box overlap (IoU)
- Detection confidence
- Object size and area
- Frame-to-frame movement

## Labels

The main task is binary collision classification:

```text
0 = No collision
1 = Collision
```

Near-miss events are also recorded separately.

## Evaluation

The evaluation produces:

```text
results/metrics.json
results/predictions.csv
results/feature_importance.csv
results/charts/
```

The main evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Data Split

The data is split chronologically rather than randomly.

This prevents nearby video frames from appearing in both training and test sets and provides a more realistic evaluation of model performance.

## Quick Start

After activating the virtual environment, run:

```powershell
python scripts\01_convert_annotations.py
python scripts\02_extract_features.py --sample-fps 5
python scripts\03_train_classifier.py
python scripts\04_evaluate.py
```

## Quick Video Test

To view the YOLO detections in the video:

```powershell
python scripts\05_visualize_detections.py
```

To start from 48 seconds:

```powershell
python scripts\05_visualize_detections.py --start-sec 48
```

## Notes

- Run the pipeline scripts in order.
- Make sure `data/video.mp4` is available before running feature extraction.
- The YOLO model is used for detecting TIPL, TIPR, and TIPandCircle.
- The XGBoost model performs the final collision classification.
