import os
import cv2
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import choice
from collections import defaultdict
from typing import Dict, Any


# ==============================
# POSTPROCESSING UTILITIES
# ==============================

def apply_dense_crf(image: np.ndarray, prob_map: np.ndarray, n_iters: int = 10) -> np.ndarray:
    """
    Apply DenseCRF post-processing to refine segmentation predictions.

    Args:
        image (np.ndarray): Original image (H, W, 3).
        prob_map (np.ndarray): Model output probability map (C, H, W).
        n_iters (int): Number of inference iterations.

    Returns:
        np.ndarray: Refined segmentation mask.
    """
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as utils

    H, W, _ = image.shape
    n_labels = prob_map.shape[0]

    d = dcrf.DenseCRF2D(W, H, n_labels)
    unary = utils.unary_from_softmax(prob_map)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)

    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape((H, W))


def apply_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological closing to remove small noise.

    Args:
        mask (np.ndarray): Binary mask.
        kernel_size (int): Size of the structuring element.

    Returns:
        np.ndarray: Smoothed mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(segmentation_mask: np.ndarray, min_size: int = 250) -> np.ndarray:
    """
    Remove small connected components from a multi-class segmentation mask.

    Args:
        segmentation_mask (np.ndarray): Multi-class segmentation mask (H, W).
        min_size (int): Minimum size of regions to keep.

    Returns:
        np.ndarray: Filtered segmentation mask with small regions removed.
    """
    unique_classes = np.unique(segmentation_mask)
    filtered_mask = np.zeros_like(segmentation_mask)

    for cls in unique_classes:
        if cls == 0:  # Ignore background
            continue

        class_mask = (segmentation_mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # Skip background label 0
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_mask[labels == i] = cls

    return filtered_mask


# ==============================
# PIPELINES
# ==============================

def process_mask(
    dt_mask: np.ndarray,
    roofs_masks: Dict[str, np.ndarray],
    roofs_ids: Dict[str, Any],
    kernel_size: int = 3,
    min_size: int = 250
) -> np.ndarray:
    """
    Process segmentation masks for different roof types.

    Args:
        dt_mask (np.ndarray): Detected mask.
        roofs_masks (dict): Dict containing roof masks per type.
        roofs_ids (dict): Dict containing class IDs per roof type.
        kernel_size (int): Kernel size for closing.
        min_size (int): Minimum connected component size.

    Returns:
        np.ndarray: Final processed mask.
    """
    ardoise_ids, autres_ids, tuiles_ids = roofs_ids["ardoise"], roofs_ids["autres"], roofs_ids["tuiles"]
    ardoise_mask, autres_mask, tuiles_mask = roofs_masks["ardoise_mask"], roofs_masks["autres_mask"], roofs_masks["tuiles_mask"]

    # Step 1: Isolate roof type predictions
    tuiles_dt = np.where(np.isin(dt_mask, tuiles_ids), dt_mask, tuiles_mask)
    ardoise_dt = np.where(np.isin(dt_mask, ardoise_ids), dt_mask, ardoise_mask)
    autres_dt = np.where(np.isin(dt_mask, autres_ids), dt_mask, autres_mask)

    # Step 2: Closing
    tuiles_closed = apply_closing(tuiles_dt, kernel_size)
    ardoise_closed = apply_closing(ardoise_dt, kernel_size)
    autres_closed = apply_closing(autres_dt, kernel_size)

    # Step 3: Remove small components
    tuiles_clean = remove_small_components(tuiles_closed, min_size)
    ardoise_clean = remove_small_components(ardoise_closed, min_size)
    autres_clean = remove_small_components(autres_closed, min_size)

    # Step 4: Enforce roof mask constraints
    finished_tuiles = np.where(~np.isin(tuiles_clean, tuiles_ids), tuiles_mask, tuiles_clean) * (tuiles_mask != 0)
    finished_ardoise = np.where(~np.isin(ardoise_clean, ardoise_ids), ardoise_mask, ardoise_clean) * (ardoise_mask != 0)
    finished_autres = np.where(~np.isin(autres_clean, autres_ids), autres_mask, autres_clean) * (autres_mask != 0)

    # Step 5: Merge
    return finished_ardoise | finished_autres | finished_tuiles


def process_pleiade(dt_mask: np.ndarray, kernel_size: int = 9, min_size: int = 1000) -> np.ndarray:
    """
    Process Pleiade segmentation mask.

    Args:
        dt_mask (np.ndarray): Segmentation mask.
        kernel_size (int): Kernel size for closing.
        min_size (int): Minimum connected component size.

    Returns:
        np.ndarray: Final processed mask.
    """
    closed = apply_closing(dt_mask, kernel_size=kernel_size)
    return remove_small_components(closed, min_size=min_size)


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    # Example usage (placeholder demo)
    print("Running demo with fake mask...")

    # Fake demo mask
    fake_mask = np.zeros((100, 100), dtype=np.uint8)
    fake_mask[20:40, 20:40] = 1
    fake_mask[60:62, 60:62] = 2  # very small component

    roofs_masks = {
        "ardoise_mask": np.zeros_like(fake_mask),
        "autres_mask": np.zeros_like(fake_mask),
        "tuiles_mask": np.zeros_like(fake_mask),
    }
    roofs_ids = {
        "ardoise": [1],
        "autres": [2],
        "tuiles": [3],
    }

    processed = process_mask(fake_mask, roofs_masks, roofs_ids)
    pleiade_processed = process_pleiade(fake_mask)

    print("Processed mask unique values:", np.unique(processed))
    print("Pleiade processed mask unique values:", np.unique(pleiade_processed))
