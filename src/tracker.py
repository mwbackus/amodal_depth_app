"""
Stage 3: ByteTrack-style multi-object tracker with Kalman filtering.
Supports both modal (standard) and amodal (occlusion-aware) modes.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """Track a single object with a Kalman filter over its bounding box."""

    _count = 0

    def __init__(self, bbox, obj_class=None):
        """Initialize with bounding box [x1, y1, x2, y2]."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State: [cx, cy, s, r, dcx, dcy, ds]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        z = self._bbox_to_z(bbox)
        self.kf.x[:4] = z.reshape(4, 1)

        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.hits = 1
        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 0
        self.obj_class = obj_class

        self.is_occluded = False
        self.occluded_frames = 0

    @staticmethod
    def _bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] to [cx, cy, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r])

    @staticmethod
    def _z_to_bbox(z):
        """Convert [cx, cy, s, r] to [x1, y1, x2, y2]."""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = max(z[2] / max(w, 1e-6), 0)
        return np.array([
            z[0] - w / 2, z[1] - h / 2,
            z[0] + w / 2, z[1] + h / 2
        ])

    def predict(self):
        """Advance state and return predicted bbox."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x.flatten()[:4])

    def update(self, bbox):
        """Update with matched detection."""
        z = self._bbox_to_z(bbox)
        self.kf.update(z.reshape(4, 1))
        self.hits += 1
        self.time_since_update = 0
        self.hit_streak += 1
        self.is_occluded = False
        self.occluded_frames = 0

    def get_state(self):
        """Return current bbox estimate."""
        return self._z_to_bbox(self.kf.x.flatten()[:4])


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of boxes. Returns (N, M) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


class MultiObjectTracker:
    """
    ByteTrack-style multi-object tracker.

    Parameters:
        max_age: frames to keep a track alive without detection
        min_hits: minimum detections before a track is confirmed
        iou_threshold: minimum IoU for matching
        use_amodal: if True, use occlusion-aware track management
        occlusion_patience: frames to keep occluded track alive (amodal mode)
    """

    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3,
                 use_amodal=False, occlusion_patience=15):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_amodal = use_amodal
        self.occlusion_patience = occlusion_patience
        self.tracks = []
        self.frame_count = 0

        self.id_switches = 0
        self.total_tracks_created = 0
        self.track_history = {}

    def update(self, detections, depth_info=None):
        """
        Update tracks with new detections.

        Args:
            detections: dict with 'boxes' (N,4) and optionally 'classes' (N,)
            depth_info: dict with 'layers' and 'graph' from depth ordering

        Returns:
            list of (track_id, bbox, is_occluded) for active tracks
        """
        self.frame_count += 1
        boxes = detections.get('boxes', np.zeros((0, 4)))
        classes = detections.get('classes', np.array([]))

        # Predict new locations
        predicted_boxes = []
        for track in self.tracks:
            pred = track.predict()
            predicted_boxes.append(pred)
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.zeros((0, 4))

        # Match predictions to detections
        if len(predicted_boxes) > 0 and len(boxes) > 0:
            iou_matrix = compute_iou_matrix(predicted_boxes, boxes)
            cost_matrix = 1.0 - iou_matrix
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()
            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    self.tracks[t_idx].update(boxes[d_idx])
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)
        else:
            matched_tracks = set()
            matched_dets = set()

        # Handle unmatched tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                if self.use_amodal and depth_info is not None:
                    # Check if predicted position overlaps with a frontward object
                    pred_bbox = predicted_boxes[t_idx] if t_idx < len(predicted_boxes) else None
                    is_likely_occluded = False

                    if pred_bbox is not None and len(boxes) > 0:
                        layers = depth_info.get('layers', np.array([]))
                        for d_idx in range(len(boxes)):
                            if d_idx < len(layers) and layers[d_idx] == 0:
                                iou = compute_iou_matrix(
                                    pred_bbox.reshape(1, 4),
                                    boxes[d_idx].reshape(1, 4)
                                )[0, 0]
                                if iou > 0.05:
                                    is_likely_occluded = True
                                    break

                    if is_likely_occluded:
                        track.is_occluded = True
                        track.occluded_frames += 1

        # Create new tracks for unmatched detections
        for d_idx in range(len(boxes)):
            if d_idx not in matched_dets:
                cls = classes[d_idx] if d_idx < len(classes) else None
                new_track = KalmanBoxTracker(boxes[d_idx], obj_class=cls)
                self.tracks.append(new_track)
                self.total_tracks_created += 1
                self.track_history[new_track.id] = []

        # Remove dead tracks
        active_tracks = []
        for track in self.tracks:
            max_patience = self.max_age
            if self.use_amodal and track.is_occluded:
                max_patience = self.occlusion_patience
            if track.time_since_update <= max_patience:
                active_tracks.append(track)
        self.tracks = active_tracks

        # Return current state
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = track.get_state()
                results.append((track.id, bbox, track.is_occluded))
                if track.id not in self.track_history:
                    self.track_history[track.id] = []
                self.track_history[track.id].append(
                    (self.frame_count, bbox.copy(), track.is_occluded)
                )

        return results
