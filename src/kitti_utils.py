"""
KITTI Tracking dataset label parsing and exploration utilities.
"""
import os
import numpy as np
from collections import defaultdict
from .config import TRACKED_CLASSES, LBL_BASE, SEQ_IDS


def parse_kitti_tracking_labels(label_file):
    """
    Parse a KITTI tracking label file into a structured dict.

    Returns:
        dict: frame_id -> list of object dicts with keys:
              frame, track_id, type, truncated, occluded, bbox
    """
    labels = defaultdict(list)
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            truncated = float(parts[3])
            occluded = int(parts[4])
            bbox = [float(parts[6]), float(parts[7]),
                    float(parts[8]), float(parts[9])]

            if obj_type in TRACKED_CLASSES:
                labels[frame_id].append({
                    'frame': frame_id,
                    'track_id': track_id,
                    'type': obj_type,
                    'truncated': truncated,
                    'occluded': occluded,
                    'bbox': bbox,
                })
    return labels


def load_all_gt_labels(seq_ids=None, lbl_base=None):
    """Load ground truth labels for all sequences."""
    if seq_ids is None:
        seq_ids = SEQ_IDS
    if lbl_base is None:
        lbl_base = LBL_BASE

    gt_labels = {}
    for seq in seq_ids:
        lbl_file = os.path.join(str(lbl_base), f"{seq}.txt")
        if os.path.exists(lbl_file):
            gt_labels[seq] = parse_kitti_tracking_labels(lbl_file)
        else:
            print(f"  WARNING: Label file not found: {lbl_file}")
    return gt_labels


def find_occlusion_events(gt_labels, seq_ids=None):
    """
    Identify occlusion events: frames where a track transitions from
    visible (occ<=1) to heavily occluded (occ>=2) or disappears.
    """
    if seq_ids is None:
        seq_ids = list(gt_labels.keys())

    occlusion_events = {}
    for seq in seq_ids:
        labels = gt_labels[seq]
        track_timeline = defaultdict(dict)
        for fid, objs in labels.items():
            for obj in objs:
                track_timeline[obj['track_id']][fid] = obj['occluded']

        events = []
        for tid, timeline in track_timeline.items():
            frames_sorted = sorted(timeline.keys())
            for i in range(1, len(frames_sorted)):
                prev_f = frames_sorted[i - 1]
                curr_f = frames_sorted[i]
                prev_occ = timeline[prev_f]
                curr_occ = timeline[curr_f]

                if prev_occ <= 1 and curr_occ >= 2:
                    events.append((tid, prev_f, curr_f, "visible->occluded"))
                if curr_f - prev_f > 3 and prev_occ <= 1:
                    events.append((tid, prev_f, curr_f, "disappeared"))

        occlusion_events[seq] = events
    return occlusion_events
