# Amodal Perception for Depth Ordering and Occlusion-Aware Tracking

**STAT 550 -- Project 2: Amodal Scene Understanding**

## Project Summary

This project implements a three-stage pipeline that uses amodal perception to reason about scene
layout over time. Starting from amodal segmentation (predicting complete object shapes even under
occlusion), we infer relative depth ordering between objects using geometric mask analysis, and
then feed this information into a multi-object tracker that maintains object identity through
complete occlusion events. The pipeline is evaluated on KITTI Tracking sequences and compared
against a modal (visible-only) detection baseline.

The key insight is that comparing amodal masks (full object extent) with visible masks (what the
camera actually sees) reveals which objects are in front of and behind each other. This occlusion
graph is converted to depth layers via topological sorting, and the tracker uses this depth
information to distinguish between "the object left the scene" and "the object went behind
another object" -- enabling longer, more consistent tracks during occlusion events.

## Pipeline Architecture

```
Video Frames
    |
    v
[Stage 1: Detection & Segmentation]
    |                    |
    v                    v
AISFormer            Faster R-CNN
(Amodal masks)       (Modal boxes only)
    |                    |
    v                    |
[Stage 2: Depth Ordering]     |
(Occlusion graph +            |
 topological sort)            |
    |                         |
    v                         v
[Stage 3: Tracking]     [Stage 3: Tracking]
Amodal-aware            Modal baseline
(occlusion patience)    (standard max_age)
    |                         |
    v                         v
[Evaluation: MOTA, IDF1, ID switches, occlusion survival]
```

## Selected Models and Datasets

| Component | Model/Dataset | Source |
|-----------|--------------|--------|
| Amodal Segmentation | AISFormer (ResNet-50-FPN) | KINS-pretrained, from [UARK-AICV/AISFormer](https://github.com/UARK-AICV/AISFormer) |
| Modal Baseline | Faster R-CNN (ResNet-50-FPN) | COCO-pretrained, from Detectron2 model zoo |
| Dataset | KITTI Tracking | 5 sequences: 0001, 0004, 0011, 0013, 0015 |
| Tracker | ByteTrack-style (Kalman + Hungarian) | Custom implementation |

## Repository Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_environment.py      # Clone AISFormer, download weights & KITTI
├── run_pipeline.py           # Full amodal pipeline (end-to-end)
├── run_baseline.py           # Modal-only baseline for comparison
├── evaluate.py               # Compute tracking metrics
├── notebooks/
│   └── mb_STAT550_..._start.ipynb      # Clean notebook (no execution, run on Colab)
├── src/
│   ├── config.py             # Paths, constants, thresholds
│   ├── kitti_utils.py        # KITTI label parsing
│   ├── inference.py          # Model loading & per-frame inference
│   ├── depth_ordering.py     # Occlusion graph & topological sort
│   ├── tracker.py            # Kalman filter & ByteTrack MOT
│   └── visualization.py      # Plotting utilities
├── configs/
│   └── pipeline.yaml         # Pipeline configuration
├── results/                  # Saved outputs from execution
│   └── summary.json
└── data/
    └── README_data.md        # Data download instructions
```

## Instructions to Reproduce All Results

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with 16+ GB memory (tested on NVIDIA 5080)
- ~15 GB disk space for KITTI data + model weights

### Option 1: Google Colab (Recommended)

1. Open `notebooks/mb_STAT550_Proj2_Amodal_Depth_Tracking_start.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Run all cells sequentially (~30-45 minutes)
4. All visualizations and metrics are generated inline

### Option 2: Local Execution (requires CUDA GPU)

```bash
# 1. Clone this repository
git clone <repo-url>
cd mb_STAT550_Proj2_Amodal_Depth_Tracking

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download AISFormer, KITTI data, and model weights
python setup_environment.py

# 5. Run the full amodal pipeline
python run_pipeline.py

# 6. Run the modal baseline for comparison
python run_baseline.py

# 7. Evaluate tracking metrics
python evaluate.py
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MOTA** | Multi-Object Tracking Accuracy (combines FP, FN, ID switches) |
| **IDF1** | ID F1 Score (measures identity preservation) |
| **Track Fragmentation** | Total tracks created vs. GT objects (fewer = better) |
| **Avg Track Length** | Mean frames per track (longer = more consistent) |
| **Occlusion Survival** | Fraction of occlusion events where track ID was maintained |
| **ID Switches** | Number of times a GT object's tracker ID changed |

## Key Findings

1. Amodal masks enable geometric depth reasoning -- comparing full vs. visible masks reveals
   which objects are in front/behind each other
2. The amodal-aware tracker reduces ID switches at occlusion events compared to the modal
   baseline by using depth information to distinguish "occluded" from "gone"
3. Error propagation is measurable: poor amodal mask quality at Stage 1 correlates with
   incorrect depth ordering at Stage 2 and tracking failures at Stage 3
