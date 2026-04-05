"""
Run the modal (standard detector) baseline pipeline for comparison.
Uses Faster R-CNN only (no amodal masks, no depth ordering).

Usage:
    python run_baseline.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SEQ_IDS, MAX_FRAMES, IMG_BASE, RESULTS_DIR,
    MODAL_MAX_AGE, MIN_HITS, IOU_THRESHOLD
)
from src.kitti_utils import load_all_gt_labels
from src.inference import load_modal_model, run_modal_inference
from src.tracker import KalmanBoxTracker, MultiObjectTracker

import cv2
import time


def main():
    print("=" * 60)
    print("MODAL BASELINE PIPELINE (Faster R-CNN only)")
    print("=" * 60)

    # Load GT
    print("\n[1/3] Loading ground truth labels...")
    gt_labels = load_all_gt_labels()

    # Load model
    print("\n[2/3] Loading Faster R-CNN...")
    modal_predictor = load_modal_model()

    # Run inference + tracking
    print("\n[3/3] Running inference and tracking...")
    modal_tracking = {}
    modal_trackers = {}

    for seq in SEQ_IDS:
        img_dir = os.path.join(str(IMG_BASE), seq)
        if not os.path.isdir(img_dir):
            print(f"  WARNING: {img_dir} not found, skipping")
            continue

        frame_files = sorted(os.listdir(img_dir))[:MAX_FRAMES]
        print(f"\n  Seq {seq} ({len(frame_files)} frames):")

        KalmanBoxTracker._count = 0
        tracker = MultiObjectTracker(
            max_age=MODAL_MAX_AGE, min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD, use_amodal=False
        )

        modal_tracking[seq] = {}
        t0 = time.time()

        for i, fname in enumerate(frame_files):
            frame_id = int(fname.split('.')[0])
            img_bgr = cv2.imread(os.path.join(img_dir, fname))
            if img_bgr is None:
                continue

            mres = run_modal_inference(modal_predictor, img_bgr)
            modal_out = tracker.update(mres)
            modal_tracking[seq][frame_id] = modal_out

            if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                fps = (i + 1) / (time.time() - t0)
                print(f"    Frame {frame_id:4d} | {fps:.1f} fps")

        modal_trackers[seq] = tracker
        print(f"    Tracks created: {tracker.total_tracks_created}")

    # Save
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    summary = {
        'pipeline': 'modal_baseline',
        'sequences': SEQ_IDS,
        'results': {seq: {'tracks_created': modal_trackers[seq].total_tracks_created}
                    for seq in modal_trackers}
    }
    out_path = os.path.join(str(RESULTS_DIR), "baseline_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
