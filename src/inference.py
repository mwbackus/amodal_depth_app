"""
Model loading and per-frame inference for AISFormer (amodal) and Faster R-CNN (modal).
"""
import os
import sys
import time
import numpy as np
import torch
import cv2
from collections import defaultdict

from .config import (
    AISFORMER_DIR, AISFORMER_WEIGHTS, AISFORMER_CONFIG,
    SCORE_THRESH, MIN_SIZE_TEST, MAX_SIZE_TEST,
    COCO_CLASSES, DRIVING_CLASSES, KINS_CLASSES,
    IMG_BASE, SEQ_IDS, MAX_FRAMES
)


def setup_aisformer_path():
    """Add AISFormer to sys.path and patch compatibility issues."""
    aisformer_path = str(AISFORMER_DIR)
    if aisformer_path not in sys.path:
        sys.path.insert(0, aisformer_path)

    # Patch collections.Container -> collections.abc.Container for Python 3.10+
    builtin_candidates = [
        os.path.join(aisformer_path, "detectron2", "config", "compat.py"),
        os.path.join(aisformer_path, "adet", "config.py"),
    ]
    for fpath in builtin_candidates:
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                old = 'from collections import Container'
                if old in content and 'collections.abc' not in content:
                    content = content.replace(old,
                                              'from collections.abc import Container')
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  Patched {os.path.basename(fpath)} for Python 3.10+ compat")
            except Exception:
                pass


def patch_pil_compat():
    """Fix PIL compatibility for Pillow 10+ (removed Image.LINEAR)."""
    from PIL import Image
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.BILINEAR


def add_aisformer_config(cfg):
    """Register AISFormer-specific config keys."""
    from detectron2.config import CfgNode as CN

    cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME = ""
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET = CN()
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME = ""
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NUM_CONVS_INSTANCE = 0
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MASK_OUT_CHANNELS = 256
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.AMODAL_CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.VISIBLE_CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.MASK_FCN_INPUT_CHANNELS = 256
    cfg.MODEL.ROI_MASK_HEAD.NUM_MASK_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.NHEAD = 8
    cfg.MODEL.ROI_MASK_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_DEC_LAYERS = 4
    cfg.MODEL.ROI_MASK_HEAD.PRE_NORM = False
    cfg.MODEL.ROI_MASK_HEAD.MASK_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.ROI_MASK_HEAD.AMODAL_CYCLE = False
    cfg.MODEL.AISFormer = CN()
    cfg.MODEL.AISFormer.USE = False
    cfg.MODEL.AISFormer.AMODAL_EVAL = False
    cfg.MODEL.AISFormer.JUSTIFY_LOSS = False
    cfg.MODEL.AISFormer.N_HEADS = 8
    cfg.MODEL.AISFormer.N_LAYERS = 4
    cfg.MODEL.ALL_LAYERS_ROI_POOLING = False
    cfg.MODEL.RPN.BOUNDARY_THRESH = -1
    cfg.DICE_LOSS = False
    cfg.OUTPUT_DIR = "./output"


def load_amodal_model():
    """Load AISFormer (amodal segmentation) model."""
    setup_aisformer_path()
    patch_pil_compat()

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    orig_dir = os.getcwd()
    os.chdir(str(AISFORMER_DIR))

    cfg = get_cfg()
    add_aisformer_config(cfg)
    cfg.merge_from_file(AISFORMER_CONFIG)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.AISFormer.AMODAL_EVAL = True
    cfg.MODEL.WEIGHTS = str(AISFORMER_WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    predictor = DefaultPredictor(cfg)
    os.chdir(orig_dir)
    print("AISFormer (KINS-pretrained) loaded successfully.")
    return predictor


def load_modal_model():
    """Load Faster R-CNN (modal baseline) model."""
    setup_aisformer_path()
    patch_pil_compat()

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    add_aisformer_config(cfg)

    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.MASK_ON = False

    try:
        from detectron2 import model_zoo
        weights_url = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    except Exception:
        weights_url = ("https://dl.fbaipublicfiles.com/detectron2/"
                       "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/"
                       "137849600/model_final_f10217.pkl")

    cfg.MODEL.WEIGHTS = weights_url
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    predictor = DefaultPredictor(cfg)
    print("Faster R-CNN (COCO-pretrained, box-only) loaded successfully.")
    return predictor


def run_amodal_inference(predictor, img_bgr):
    """Run AISFormer and extract amodal + visible masks."""
    with torch.no_grad():
        outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    result = {
        'boxes': instances.pred_boxes.tensor.numpy() if len(instances) > 0
                 else np.zeros((0, 4)),
        'scores': instances.scores.numpy() if len(instances) > 0
                  else np.array([]),
        'classes': instances.pred_classes.numpy() if len(instances) > 0
                   else np.array([], dtype=int),
    }

    # AISFormer field names:
    #   pred_masks    = amodal masks (full object including occluded parts)
    #   pred_masks_bo = visible/occluder masks
    if len(instances) > 0 and hasattr(instances, 'pred_masks'):
        result['amodal_masks'] = instances.pred_masks.numpy()
    else:
        result['amodal_masks'] = np.zeros((0, 1, 1), dtype=bool)

    if len(instances) > 0 and hasattr(instances, 'pred_masks_bo'):
        result['visible_masks'] = instances.pred_masks_bo.numpy()
    else:
        result['visible_masks'] = result['amodal_masks'].copy()

    return result


def run_modal_inference(predictor, img_bgr):
    """Run Faster R-CNN and extract visible boxes only."""
    with torch.no_grad():
        outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy() if len(instances) > 0 else np.array([], dtype=int)
    pred_scores = instances.scores.numpy() if len(instances) > 0 else np.array([])
    pred_boxes = instances.pred_boxes.tensor.numpy() if len(instances) > 0 else np.zeros((0, 4))

    if len(pred_classes) > 0:
        keep = np.array([COCO_CLASSES[c] in DRIVING_CLASSES for c in pred_classes])
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        pred_boxes = pred_boxes[keep]
        class_names = [COCO_CLASSES[c] for c in pred_classes]
    else:
        class_names = []

    return {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'classes': pred_classes,
        'class_names': class_names,
    }


def run_all_inference(amodal_predictor, modal_predictor, seq_ids=None,
                      max_frames=None, img_base=None):
    """Run both models on all sequences. Returns amodal_results, modal_results dicts."""
    if seq_ids is None:
        seq_ids = SEQ_IDS
    if max_frames is None:
        max_frames = MAX_FRAMES
    if img_base is None:
        img_base = str(IMG_BASE)

    amodal_results = {}
    modal_results = {}
    total_frames_processed = 0
    t_start = time.time()

    for seq in seq_ids:
        img_dir = os.path.join(img_base, seq)
        if not os.path.isdir(img_dir):
            print(f"  WARNING: Image directory not found: {img_dir}")
            continue
        frame_files = sorted(os.listdir(img_dir))[:max_frames]
        n_frames = len(frame_files)
        print(f"\nSequence {seq} ({n_frames} frames):")
        seq_t0 = time.time()

        for i, fname in enumerate(frame_files):
            frame_id = int(fname.split('.')[0])
            img_path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                print(f"  WARNING: Could not read {img_path}, skipping")
                continue

            amodal_results[(seq, frame_id)] = run_amodal_inference(amodal_predictor, img_bgr)
            modal_results[(seq, frame_id)] = run_modal_inference(modal_predictor, img_bgr)
            total_frames_processed += 1

            if (i + 1) % 30 == 0 or i == n_frames - 1:
                elapsed = time.time() - seq_t0
                fps = (i + 1) / elapsed
                n_amodal = len(amodal_results[(seq, frame_id)]['boxes'])
                n_modal = len(modal_results[(seq, frame_id)]['boxes'])
                print(f"  Frame {frame_id:4d}  |  amodal: {n_amodal:2d} det  |  "
                      f"modal: {n_modal:2d} det  |  {fps:.1f} fps")

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total frames:    {total_frames_processed}")
    print(f"  Total time:      {total_time:.0f}s ({total_time / 60:.1f} min)")
    if total_time > 0:
        print(f"  Avg speed:       {total_frames_processed / total_time:.1f} frames/sec")

    return amodal_results, modal_results, total_frames_processed
