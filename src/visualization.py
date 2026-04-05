"""
Visualization utilities for depth layers, track overlays, and comparisons.
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from .config import KINS_CLASSES, IMG_BASE


DEPTH_CMAP = cm.get_cmap('RdYlBu_r', 6)


def get_track_color(track_id, n_colors=20):
    """Return a consistent RGB color for a track ID."""
    cmap = plt.cm.get_cmap('tab20', n_colors)
    return cmap(track_id % n_colors)[:3]


def visualize_depth_layers(seq, frame_id, amodal_results, depth_results,
                           img_base=None, ax=None):
    """Draw depth-colored masks on a frame."""
    if img_base is None:
        img_base = str(IMG_BASE)

    key = (seq, frame_id)
    if key not in amodal_results or key not in depth_results:
        return

    ares = amodal_results[key]
    dres = depth_results[key]
    layers = dres['layers']

    img_path = os.path.join(img_base, seq, f"{frame_id:06d}.png")
    img = cv2.imread(img_path)
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    ax.imshow(img_rgb)

    from .depth_ordering import masks_to_image_size
    amodal_masks = masks_to_image_size(ares['amodal_masks'], img_h, img_w)
    max_layer = max(layers.max(), 1) if len(layers) > 0 else 1

    for i, (mask, layer) in enumerate(zip(amodal_masks, layers)):
        color = DEPTH_CMAP(layer / max_layer)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask, :] = (*color[:3], 0.35)
        ax.imshow(overlay)

        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_squeezed = cnt.squeeze()
            if cnt_squeezed.ndim == 2 and len(cnt_squeezed) > 2:
                ax.plot(cnt_squeezed[:, 0], cnt_squeezed[:, 1],
                        color=color[:3], linewidth=2)

        if i < len(ares['boxes']):
            box = ares['boxes'][i]
            cls_id = ares['classes'][i]
            cls_name = KINS_CLASSES.get(cls_id, f"cls{cls_id}")
            ax.text(box[0], box[1] - 5, f"{cls_name} L{layer}",
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                             facecolor=color[:3], alpha=0.8))

    ax.set_title(f"Seq {seq}, Frame {frame_id} "
                 f"({len(layers)} objects, {dres['max_layer'] + 1} depth layers)",
                 fontsize=11)
    ax.axis('off')


def draw_tracks_on_frame(img_rgb, tracks, title="", ax=None, track_history=None):
    """
    Draw tracked bounding boxes and trajectory trails on a frame.

    Args:
        img_rgb: RGB image array
        tracks: list of (track_id, bbox, is_occluded)
        title: title for the subplot
        ax: matplotlib axes
        track_history: dict mapping track_id -> list of (frame, bbox, is_occluded)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    ax.imshow(img_rgb)

    for track_id, bbox, is_occluded in tracks:
        color = get_track_color(track_id)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        linestyle = '--' if is_occluded else '-'
        alpha = 0.4 if is_occluded else 0.9
        linewidth = 1.5 if is_occluded else 2.5

        rect = mpatches.FancyBboxPatch(
            (x1, y1), w, h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=(*color, 0.1 if is_occluded else 0.05),
            linestyle=linestyle,
            boxstyle="round,pad=1"
        )
        ax.add_patch(rect)

        # Draw trajectory trail from history
        if track_history and track_id in track_history:
            history = track_history[track_id]
            if len(history) > 1:
                trail_len = min(15, len(history))
                recent = history[-trail_len:]
                centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                           for _, b, _ in recent]
                for k in range(1, len(centers)):
                    _, _, was_occluded = recent[k]
                    ls = ':' if was_occluded else '-'
                    ax.plot([centers[k-1][0], centers[k][0]],
                            [centers[k-1][1], centers[k][1]],
                            color=color, linewidth=1.5, linestyle=ls,
                            alpha=0.7)

        label = f"ID {track_id}"
        if is_occluded:
            label += " (occl)"
        ax.text(x1, y1 - 8, label,
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor=color, alpha=alpha))

    ax.set_title(title, fontsize=10)
    ax.axis('off')
