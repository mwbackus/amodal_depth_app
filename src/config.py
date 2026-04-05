"""
Central configuration for paths, constants, and model parameters.
"""
import os
from pathlib import Path

# ---- Project root ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Data paths ----
DATA_DIR = PROJECT_ROOT / "data"
KITTI_TRACK_DIR = DATA_DIR / "kitti_tracking"
IMG_BASE = KITTI_TRACK_DIR / "training" / "image_02"
LBL_BASE = KITTI_TRACK_DIR / "training" / "label_02"

# ---- AISFormer paths ----
AISFORMER_DIR = PROJECT_ROOT / "aisformer"
AISFORMER_WEIGHTS = AISFORMER_DIR / "weights" / "model_final.pth"
AISFORMER_CONFIG = "configs/KINS-AmodalSeg/aisformer_R_50_FPN_1x_amodal_kins.yaml"

# ---- Results ----
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Sequences to process ----
SEQ_IDS = ["0001", "0004", "0011", "0013", "0015"]
MAX_FRAMES = 150

# ---- Model thresholds ----
SCORE_THRESH = 0.5
IOU_THRESHOLD = 0.3
MIN_SIZE_TEST = 800
MAX_SIZE_TEST = 1333

# ---- Tracker parameters ----
MODAL_MAX_AGE = 5
AMODAL_MAX_AGE = 5
AMODAL_OCCLUSION_PATIENCE = 15
MIN_HITS = 2

# ---- Dataset classes ----
TRACKED_CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Cyclist"}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

DRIVING_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

KINS_CLASSES = {
    1: "cyclist", 2: "pedestrian", 3: "rider",
    4: "car", 5: "tram", 6: "truck", 7: "van", 8: "misc"
}

# ---- Download URLs ----
KITTI_IMAGES_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"
KITTI_LABELS_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip"
AISFORMER_WEIGHTS_GDRIVE_FOLDER_ID = "1NJhpPlbtkNBSukhT4tRPZhqHbGmvcwLI"
AISFORMER_REPO_URL = "https://github.com/UARK-AICV/AISFormer.git"
