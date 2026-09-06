# Collision Detection Project

A computer vision pipeline for detecting collisions in a pattern-cutting video.

## Project Pipeline

Video → YOLO Segmentation → Feature Extraction → XGBoost → Collision Prediction → Evaluation

## Project Structure

```text
collision_project/
├── models/
│   ├── video_tip_circle_yolo11n_seg_best.pt
│   └── xgb_collision_classifier.json
├── data/
│   ├── video.mp4
│   ├── ground_truth.csv
│   └── features_and_labels.csv
├── docs/
│   └── Pattern_Cutting_Annotation_1.docx
├── scripts/
│   ├── 01_convert_annotations.py
│   ├── 02_extract_features.py
│   ├── 03_train_classifier.py
│   ├── 04_evaluate.py
│   └── 05_visualize_detections.py
└── results/
    ├── predictions.csv
    ├── feature_importance.csv
    ├── metrics.json
    └── charts/
Environment Setup

Open PowerShell and run:

cd C:\Users\ashu1\OneDrive\Desktop\collision_project

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
Run the Complete Pipeline

Run the scripts in this order:

python scripts\01_convert_annotations.py

python scripts\02_extract_features.py --sample-fps 5

python scripts\03_train_classifier.py

python scripts\04_evaluate.py
View YOLO Detections

To open the video and see the detected objects:

python scripts\05_visualize_detections.py

To start from a specific time, for example 48 seconds:

python scripts\05_visualize_detections.py --start-sec 48
Video Controls
Space → Pause / Resume
N → Next frame
Q → Quit
Ground Truth

The annotation file is:

docs/Pattern_Cutting_Annotation_1.docx

Its timestamps use M.SS format.

Example:

3.09 = 3 minutes 9 seconds

The conversion script creates:

data/ground_truth.csv
Features

The system extracts visual features from YOLO detections, including:

TIPL/TIPR positions
Distance between tips
Distance change over time
Bounding-box overlap (IoU)
Detection confidence
Object size/area
Frame-to-frame movement
Labels

The main task is binary classification:

0 = No collision
1 = Collision

Near-miss events are also stored separately.

Evaluation

The evaluation script produces:

results/metrics.json
results/predictions.csv
results/feature_importance.csv
results/charts/

Important metrics include:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Important Note

The data is split chronologically rather than randomly. This prevents nearby video frames from appearing in both training and test data and gives a more realistic evaluation.

Quick Start

After activating the environment:

python scripts\01_convert_annotations.py
python scripts\02_extract_features.py --sample-fps 5
python scripts\03_train_classifier.py
python scripts\04_evaluate.py

To view the video detections:

python scripts\05_visualize_detections.py

### 🎥 For running the video

The **one command you need** is:

```powershell
cd C:\Users\ashu1\OneDrive\Desktop\collision_project
.\venv\Scripts\Activate.ps1
python scripts\05_visualize_detections.py

This opens the data\video.mp4 and shows the YOLO detections. Your project structure and feature-extraction approach are consistent with the pipeline you described.
