"""
Run the full amodal -> depth ordering -> tracking pipeline end-to-end.

Usage:
    python run_pipeline.py
"""
import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SEQ_IDS, MAX_FRAMES, IMG_BASE, RESULTS_DIR,
    MODAL_MAX_AGE, AMODAL_MAX_AGE, AMODAL_OCCLUSION_PATIENCE, MIN_HITS,
    IOU_THRESHOLD
)
from src.kitti_utils import load_all_gt_labels, find_occlusion_events
from src.inference import load_amodal_model, load_modal_model, run_all_inference
from src.depth_ordering import compute_all_depth_ordering
from src.tracker import KalmanBoxTracker, MultiObjectTracker


def main():
    print("=" * 60)
    print("AMODAL DEPTH-ORDERED TRACKING PIPELINE")
    print("=" * 60)

    # ---- Step 1: Load ground truth labels ----
    print("\n[1/6] Loading ground truth labels...")
    gt_labels = load_all_gt_labels()
    occlusion_events = find_occlusion_events(gt_labels)
    for seq in SEQ_IDS:
        n_events = len(occlusion_events.get(seq, []))
        print(f"  Seq {seq}: {n_events} occlusion events")

    # ---- Step 2: Load models ----
    print("\n[2/6] Loading detection models...")
    amodal_predictor = load_amodal_model()
    modal_predictor = load_modal_model()

    # ---- Step 3: Run inference ----
    print("\n[3/6] Running per-frame inference...")
    amodal_results, modal_results, total_frames = run_all_inference(
        amodal_predictor, modal_predictor
    )

    # ---- Step 4: Compute depth ordering ----
    print("\n[4/6] Computing depth/layer ordering...")
    depth_results = compute_all_depth_ordering(amodal_results)

    for seq in SEQ_IDS:
        seq_results = {k: v for k, v in depth_results.items() if k[0] == seq}
        if seq_results:
            max_layers = max(v['max_layer'] for v in seq_results.values())
            print(f"  Seq {seq}: max depth layers = {max_layers}")

    # ---- Step 5: Run tracking ----
    print("\n[5/6] Running modal and amodal trackers...")
    modal_tracking = {}
    amodal_tracking = {}
    modal_trackers = {}
    amodal_trackers = {}

    for seq in SEQ_IDS:
        KalmanBoxTracker._count = 0
        modal_tracker = MultiObjectTracker(
            max_age=MODAL_MAX_AGE, min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD, use_amodal=False
        )

        KalmanBoxTracker._count = 0
        amodal_tracker = MultiObjectTracker(
            max_age=AMODAL_MAX_AGE, min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD, use_amodal=True,
            occlusion_patience=AMODAL_OCCLUSION_PATIENCE
        )

        modal_tracking[seq] = {}
        amodal_tracking[seq] = {}

        frame_ids = sorted([fid for (s, fid) in modal_results.keys() if s == seq])

        for frame_id in frame_ids:
            mres = modal_results.get(
                (seq, frame_id),
                {'boxes': np.zeros((0, 4)), 'classes': np.array([])}
            )
            modal_out = modal_tracker.update(mres)
            modal_tracking[seq][frame_id] = modal_out

            ares = amodal_results.get(
                (seq, frame_id),
                {'boxes': np.zeros((0, 4)), 'classes': np.array([])}
            )
            dres = depth_results.get((seq, frame_id), None)
            amodal_out = amodal_tracker.update(ares, depth_info=dres)
            amodal_tracking[seq][frame_id] = amodal_out

        modal_trackers[seq] = modal_tracker
        amodal_trackers[seq] = amodal_tracker

        m_tracks = modal_tracker.total_tracks_created
        a_tracks = amodal_tracker.total_tracks_created
        print(f"  Seq {seq}: modal={m_tracks} tracks, amodal={a_tracks} tracks")

    # ---- Step 6: Save summary and frame-level tracking data ----
    print("\n[6/6] Saving results...")
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    metrics = {'seq': [], 'modal_tracks': [], 'amodal_tracks': [],
               'modal_ids': [], 'amodal_ids': [],
               'modal_avg_len': [], 'amodal_avg_len': [],
               'gt_unique_objects': []}

    for seq in SEQ_IDS:
        mt = modal_trackers[seq]
        at = amodal_trackers[seq]
        m_active_ids = set(tid for fdata in modal_tracking[seq].values()
                           for tid, _, _ in fdata)
        a_active_ids = set(tid for fdata in amodal_tracking[seq].values()
                           for tid, _, _ in fdata)

        m_track_lens = [sum(1 for fdata in modal_tracking[seq].values()
                            if any(t[0] == tid for t in fdata))
                        for tid in m_active_ids]
        a_track_lens = [sum(1 for fdata in amodal_tracking[seq].values()
                            if any(t[0] == tid for t in fdata))
                        for tid in a_active_ids]

        gt_tids = set(obj['track_id'] for objs in gt_labels[seq].values()
                      for obj in objs)

        metrics['seq'].append(seq)
        metrics['modal_tracks'].append(mt.total_tracks_created)
        metrics['amodal_tracks'].append(at.total_tracks_created)
        metrics['modal_ids'].append(len(m_active_ids))
        metrics['amodal_ids'].append(len(a_active_ids))
        metrics['modal_avg_len'].append(float(np.mean(m_track_lens)) if m_track_lens else 0)
        metrics['amodal_avg_len'].append(float(np.mean(a_track_lens)) if a_track_lens else 0)
        metrics['gt_unique_objects'].append(len(gt_tids))

    # Save frame-level tracking data for evaluate.py
    frame_level = {'gt': {}, 'modal': {}, 'amodal': {}}
    for seq in SEQ_IDS:
        frame_level['gt'][seq] = {}
        for fid, objs in gt_labels[seq].items():
            frame_level['gt'][seq][str(fid)] = [
                {'track_id': obj['track_id'],
                 'bbox': obj['bbox'].tolist() if hasattr(obj['bbox'], 'tolist')
                         else list(obj['bbox'])}
                for obj in objs
            ]
        frame_level['modal'][seq] = {}
        for fid, tracks in modal_tracking[seq].items():
            frame_level['modal'][seq][str(fid)] = [
                [int(tid), bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)]
                for tid, bbox, _ in tracks
            ]
        frame_level['amodal'][seq] = {}
        for fid, tracks in amodal_tracking[seq].items():
            frame_level['amodal'][seq][str(fid)] = [
                [int(tid), bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)]
                for tid, bbox, _ in tracks
            ]

    tracking_path = os.path.join(str(RESULTS_DIR), "tracking_data.json")
    with open(tracking_path, 'w') as f:
        json.dump(frame_level, f)

    summary = {
        'sequences_processed': SEQ_IDS,
        'frames_per_sequence': MAX_FRAMES,
        'total_frames': total_frames,
        'tracking_metrics': metrics,
    }

    summary_path = os.path.join(str(RESULTS_DIR), "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")
    print(f"Frame-level tracking: {tracking_path}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
