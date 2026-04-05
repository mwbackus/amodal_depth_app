"""
Stage 2: Occlusion-aware depth/layer ordering via geometric mask analysis.
"""
import numpy as np
import cv2
import os
from collections import defaultdict

from .config import IMG_BASE


def masks_to_image_size(masks, img_h, img_w):
    """Resize masks (N, H', W') to image dimensions (N, H, W)."""
    if len(masks) == 0:
        return np.zeros((0, img_h, img_w), dtype=bool)

    resized = []
    for m in masks:
        if m.ndim == 3:
            m = m[0]
        if m.shape[0] != img_h or m.shape[1] != img_w:
            m_resized = cv2.resize(m.astype(np.uint8), (img_w, img_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            m_resized = m.astype(np.uint8)
        resized.append(m_resized > 0)
    return np.array(resized)


def compute_depth_ordering(amodal_masks, visible_masks, min_overlap=50):
    """
    Compute relative depth ordering from amodal vs visible masks.

    For each object pair (i, j): if j's visible region overlaps with i's
    occluded region, then j is in front of i. Builds a directed graph and
    computes topological ordering for depth layers.

    Args:
        amodal_masks: (N, H, W) boolean array of complete object masks
        visible_masks: (N, H, W) boolean array of visible-only masks
        min_overlap: minimum pixel overlap to establish occlusion relationship

    Returns:
        depth_layers: array of ints, one per object (0 = frontmost)
        occlusion_graph: dict mapping occluded_idx -> set of occluder_idxs
    """
    n = len(amodal_masks)
    if n == 0:
        return np.array([], dtype=int), {}

    graph = defaultdict(set)
    behind_count = defaultdict(int)

    for i in range(n):
        occluded_i = amodal_masks[i] & (~visible_masks[i])
        occluded_area = occluded_i.sum()

        if occluded_area < min_overlap:
            continue

        for j in range(n):
            if i == j:
                continue
            overlap = (occluded_i & visible_masks[j]).sum()
            if overlap >= min_overlap:
                if j not in graph[i]:
                    graph[i].add(j)

    for occluded, occluders in graph.items():
        behind_count[occluded] = len(occluders)

    # BFS layering (topological sort)
    depth_layers = np.zeros(n, dtype=int)
    assigned = set()
    current_layer = 0
    frontier = [i for i in range(n) if behind_count[i] == 0]

    # Handle pure cycles: if every node has an occluder, frontier is empty.
    # Fall back to assigning all unassigned nodes to the deepest layer.
    if not frontier and n > 0:
        # All nodes are in cycles; assign them all to layer 0
        return depth_layers, dict(graph)

    while frontier:
        for node in frontier:
            depth_layers[node] = current_layer
            assigned.add(node)

        next_frontier = []
        for i in range(n):
            if i in assigned:
                continue
            if all(j in assigned for j in graph[i]):
                next_frontier.append(i)

        frontier = next_frontier
        current_layer += 1

        if current_layer > n:
            # Remaining nodes are in cycles; assign to deepest layer
            for i in range(n):
                if i not in assigned:
                    depth_layers[i] = current_layer
            break

    return depth_layers, dict(graph)


def compute_all_depth_ordering(amodal_results, img_base=None):
    """Compute depth ordering for all frames in amodal_results."""
    if img_base is None:
        img_base = str(IMG_BASE)

    depth_results = {}

    for (seq, frame_id), ares in amodal_results.items():
        img_dir = os.path.join(img_base, seq)
        sample_img = cv2.imread(os.path.join(img_dir, f"{frame_id:06d}.png"))
        if sample_img is None:
            print(f"  WARNING: Could not read image for depth ordering, skipping")
            continue
        img_h, img_w = sample_img.shape[:2]

        amodal_masks = masks_to_image_size(ares['amodal_masks'], img_h, img_w)
        visible_masks = masks_to_image_size(ares['visible_masks'], img_h, img_w)

        layers, graph = compute_depth_ordering(amodal_masks, visible_masks)
        depth_results[(seq, frame_id)] = {
            'layers': layers,
            'graph': graph,
            'n_objects': len(layers),
            'max_layer': int(layers.max()) if len(layers) > 0 else 0,
        }

    return depth_results
