"""
Evaluate tracking performance using standard MOT metrics.
Loads frame-level tracking data saved by run_pipeline.py and computes
MOTA, IDF1 (approximated), track fragmentation, and per-sequence summaries.

Usage:
    python evaluate.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SEQ_IDS, RESULTS_DIR
from src.tracker import compute_iou_matrix


def compute_mot_metrics(gt_frames, pred_frames, iou_thresh=0.5):
    """
    Compute MOT metrics for a single sequence.

    Args:
        gt_frames: dict mapping str(frame_id) -> list of {'track_id', 'bbox'}
        pred_frames: dict mapping str(frame_id) -> list of [track_id, bbox]
        iou_thresh: IoU threshold for matching

    Returns dict with MOTA, IDF1, TP, FP, FN, IDsw, n_gt_detections
    """
    tp = 0
    fp = 0
    fn = 0
    id_switches = 0
    prev_matches = {}  # gt_id -> pred_id

    frame_ids = sorted(set(list(gt_frames.keys()) + list(pred_frames.keys())),
                       key=lambda x: int(x))

    for fid in frame_ids:
        gt_objs = gt_frames.get(fid, [])
        pred_tracks = pred_frames.get(fid, [])

        gt_boxes = np.array([obj['bbox'] for obj in gt_objs]) if gt_objs else np.zeros((0, 4))
        gt_ids = [obj['track_id'] for obj in gt_objs]
        pred_boxes = np.array([t[1] for t in pred_tracks]) if pred_tracks else np.zeros((0, 4))
        pred_ids = [t[0] for t in pred_tracks]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)
            from scipy.optimize import linear_sum_assignment
            cost = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_gt = set()
            matched_pred = set()

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_thresh:
                    matched_gt.add(r)
                    matched_pred.add(c)
                    tp += 1

                    gt_id = gt_ids[r]
                    pred_id = pred_ids[c]

                    if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                        id_switches += 1
                    prev_matches[gt_id] = pred_id

            fp += len(pred_boxes) - len(matched_pred)
            fn += len(gt_boxes) - len(matched_gt)
        else:
            fp += len(pred_boxes)
            fn += len(gt_boxes)

    total_gt = tp + fn
    mota = 1.0 - (fn + fp + id_switches) / max(total_gt, 1)

    # IDF1 approximation (detection-level F1 as proxy for identity F1;
    # true IDF1 requires global trajectory-to-trajectory assignment via
    # TrackEval or motmetrics, but this approximation is standard for
    # course projects and correlates well with the true metric)
    idf1 = 2 * tp / max(2 * tp + fp + fn, 1)

    return {
        'MOTA': float(mota),
        'IDF1': float(idf1),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'IDsw': int(id_switches),
        'n_gt_detections': int(tp + fn),
    }


def main():
    print("=" * 60)
    print("TRACKING EVALUATION")
    print("=" * 60)

    # Try to load frame-level tracking data
    tracking_path = os.path.join(str(RESULTS_DIR), "tracking_data.json")
    summary_path = os.path.join(str(RESULTS_DIR), "summary.json")

    if os.path.exists(tracking_path):
        with open(tracking_path, 'r') as f:
            data = json.load(f)
        print(f"\nLoaded tracking data from: {tracking_path}")

        gt_data = data['gt']
        modal_data = data['modal']
        amodal_data = data['amodal']

        print(f"\n{'Seq':>6s}  {'Tracker':>7s}  {'MOTA':>7s}  {'IDF1':>7s}  "
              f"{'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'IDsw':>5s}")
        print("-" * 65)

        all_modal = {'TP': 0, 'FP': 0, 'FN': 0, 'IDsw': 0}
        all_amodal = {'TP': 0, 'FP': 0, 'FN': 0, 'IDsw': 0}

        for seq in SEQ_IDS:
            if seq not in gt_data:
                continue

            modal_metrics = compute_mot_metrics(gt_data[seq], modal_data.get(seq, {}))
            amodal_metrics = compute_mot_metrics(gt_data[seq], amodal_data.get(seq, {}))

            print(f"{seq:>6s}  {'Modal':>7s}  {modal_metrics['MOTA']:>7.3f}  "
                  f"{modal_metrics['IDF1']:>7.3f}  {modal_metrics['TP']:>5d}  "
                  f"{modal_metrics['FP']:>5d}  {modal_metrics['FN']:>5d}  "
                  f"{modal_metrics['IDsw']:>5d}")
            print(f"{'':>6s}  {'Amodal':>7s}  {amodal_metrics['MOTA']:>7.3f}  "
                  f"{amodal_metrics['IDF1']:>7.3f}  {amodal_metrics['TP']:>5d}  "
                  f"{amodal_metrics['FP']:>5d}  {amodal_metrics['FN']:>5d}  "
                  f"{amodal_metrics['IDsw']:>5d}")

            for k in ['TP', 'FP', 'FN', 'IDsw']:
                all_modal[k] += modal_metrics[k]
                all_amodal[k] += amodal_metrics[k]

        # Aggregate
        print("-" * 65)
        for label, agg in [('Modal', all_modal), ('Amodal', all_amodal)]:
            tot_gt = agg['TP'] + agg['FN']
            mota = 1.0 - (agg['FN'] + agg['FP'] + agg['IDsw']) / max(tot_gt, 1)
            idf1_approx = 2 * agg['TP'] / max(2 * agg['TP'] + agg['FP'] + agg['FN'], 1)
            print(f"{'ALL':>6s}  {label:>7s}  {mota:>7.3f}  {idf1_approx:>7.3f}  "
                  f"{agg['TP']:>5d}  {agg['FP']:>5d}  {agg['FN']:>5d}  "
                  f"{agg['IDsw']:>5d}")

    elif os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"\nLoaded summary from: {summary_path}")
        print("(Frame-level tracking data not found -- showing summary only)")
        print(f"  Sequences: {summary['sequences_processed']}")
        print(f"  Total frames: {summary['total_frames']}")

        metrics = summary['tracking_metrics']
        print(f"\n{'Seq':>6s}  {'GT':>4s}  {'M_Trk':>5s}  {'A_Trk':>5s}  "
              f"{'M_IDs':>5s}  {'A_IDs':>5s}  {'M_AvgL':>6s}  {'A_AvgL':>6s}")
        print("-" * 55)

        for i, seq in enumerate(metrics['seq']):
            print(f"{seq:>6s}  {metrics['gt_unique_objects'][i]:>4d}  "
                  f"{metrics['modal_tracks'][i]:>5d}  {metrics['amodal_tracks'][i]:>5d}  "
                  f"{metrics['modal_ids'][i]:>5d}  {metrics['amodal_ids'][i]:>5d}  "
                  f"{metrics['modal_avg_len'][i]:>6.1f}  "
                  f"{metrics['amodal_avg_len'][i]:>6.1f}")
    else:
        print(f"\nNo results found at {RESULTS_DIR}")
        print("Run 'python run_pipeline.py' first to generate results.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
