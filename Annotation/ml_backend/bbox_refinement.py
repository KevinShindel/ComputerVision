"""Bounding box refinement using segmentation masks.

Improves YOLO bounding boxes by computing the actual bounding box from SAM2 masks.
This helps correct for YOLO's inaccurate detections.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_bbox_from_mask(mask: np.ndarray) -> tuple | None:
    """Compute bounding box from a binary segmentation mask.

    Args:
        mask: Binary mask (0 or 1) or float mask (0.0 to 1.0)

    Returns:
        (x1, y1, x2, y2) in absolute pixels, or None if mask is empty
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask_u8 = (mask * 255).astype(np.uint8)
    else:
        mask_u8 = mask

    # Find contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get bounding box of largest contour (the actual object)
    largest_contour = max(contours, key=cv2.contourArea)
    x1, y1, w, h = cv2.boundingRect(largest_contour)
    x2, y2 = x1 + w, y1 + h

    return (x1, y1, x2, y2)


def compute_mask_centroid(mask: np.ndarray) -> tuple | None:
    """Compute the weighted center of mass of a segmentation mask.

    Args:
        mask: Binary mask (0 or 1) or float mask (0.0 to 1.0)

    Returns:
        (cx, cy) center coordinates, or None if mask is empty
    """
    if mask.dtype != np.uint8:
        mask_u8 = (mask * 255).astype(np.uint8)
    else:
        mask_u8 = mask

    # Compute moments
    moments = cv2.moments(mask_u8)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return (cx, cy)


def refine_bbox_with_mask(
    yolo_bbox: tuple, mask: np.ndarray, use_centroid: bool = False
) -> tuple:
    """Refine YOLO bounding box using SAM2 segmentation mask.

    This can use two approaches:
    1. Compute bbox from mask contour (more precise)
    2. Use mask centroid to adjust bbox center (subtle adjustment)

    Args:
        yolo_bbox: (x1, y1, x2, y2) from YOLO
        mask: SAM2 segmentation mask
        use_centroid: If True, adjust bbox center using mask centroid.
                     If False, compute bbox directly from mask.

    Returns:
        Refined (x1, y1, x2, y2) tuple
    """
    if use_centroid:
        # Subtle approach: adjust YOLO bbox center to mask centroid
        centroid = compute_mask_centroid(mask)
        if centroid is None:
            return yolo_bbox

        cx_mask, cy_mask = centroid
        x1, y1, x2, y2 = yolo_bbox

        # Compute YOLO bbox center
        cx_yolo = (x1 + x2) / 2
        cy_yolo = (y1 + y2) / 2

        # Compute offset
        dx = cx_mask - cx_yolo
        dy = cy_mask - cy_yolo

        # Apply offset (smaller adjustment for stability)
        # Use 70% of the offset to avoid overcorrection
        x1_refined = x1 + dx * 0.7
        y1_refined = y1 + dy * 0.7
        x2_refined = x2 + dx * 0.7
        y2_refined = y2 + dy * 0.7

        logger.debug(f"Centroid refinement: offset=({dx:.1f}, {dy:.1f})")
        return (x1_refined, y1_refined, x2_refined, y2_refined)
    else:
        # Direct approach: compute bbox from mask
        mask_bbox = compute_bbox_from_mask(mask)
        if mask_bbox is None:
            return yolo_bbox

        logger.debug(f"Mask bbox: {mask_bbox}, YOLO bbox: {yolo_bbox}")
        return mask_bbox


def refine_bboxes_batch(
    yolo_bboxes: np.ndarray, masks: np.ndarray, use_centroid: bool = False
) -> np.ndarray:
    """Refine multiple YOLO bounding boxes using SAM2 masks.

    Args:
        yolo_bboxes: Array of shape [N, 4] with (x1, y1, x2, y2)
        masks: Array of shape [N, H, W] with binary masks
        use_centroid: Refinement strategy (see refine_bbox_with_mask)

    Returns:
        Refined bboxes array of shape [N, 4]
    """
    refined = yolo_bboxes.copy()

    for i, (bbox, mask) in enumerate(zip(yolo_bboxes, masks)):
        refined_bbox = refine_bbox_with_mask(bbox, mask, use_centroid=use_centroid)
        refined[i] = refined_bbox

    return refined
