"""
YOLO26 + SAM2 Label Studio ML Backend with SAM-based bbox refinement.

Pipeline:
  1. YOLO26 detects objects → bounding boxes + class labels
  2. SAM2 uses those boxes as prompts → pixel-level masks per object
  3. Masks refine bounding box coordinates (corrects YOLO inaccuracies)
  4. Results are returned as RectangleLabels with refined coordinates
"""

import logging
import os

import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image
from ultralytics import SAM, YOLO

from bbox_refinement import refine_bboxes_batch

logger = logging.getLogger(__name__)


def _resolve_model_path(env_var: str, default: str) -> str:
    """Resolve a model path, checking the env var and validating the file exists."""
    path = os.getenv(env_var, default)
    if not os.path.isabs(path):
        # Resolve relative to the directory containing this file
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path!r} (set {env_var} to override)"
        )
    return path


class YoloSamBackend(LabelStudioMLBase):
    """Label Studio ML backend: YOLO26 detection + SAM2 bbox refinement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        yolo_path = _resolve_model_path("YOLO_MODEL", "models/yolo26s.pt")
        sam_path = _resolve_model_path("SAM_MODEL", "models/sam2.1_b.pt")
        self.conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.25"))
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.45"))
        # "mask"     → compute bbox directly from SAM2 mask contour (more precise)
        # "centroid" → shift YOLO bbox center to mask centroid (subtle correction)
        self.refinement_strategy = os.getenv("REFINEMENT_STRATEGY", "mask")

        # Label Studio annotation tag names — must match your labeling config XML
        self.bbox_from_name = os.getenv("BBOX_FROM_NAME", "bbox_labels")
        self.image_to_name = os.getenv("IMAGE_TO_NAME", "image")
        self.image_value = os.getenv("IMAGE_VALUE", "image")

        logger.info("Loading YOLO model: %s", yolo_path)
        self.yolo = YOLO(yolo_path)

        logger.info("Loading SAM model: %s", sam_path)
        self.sam = SAM(sam_path)

        logger.info(
            "YoloSamBackend ready | strategy=%s | conf=%.2f | iou=%.2f",
            self.refinement_strategy, self.conf_threshold, self.iou_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"].get(self.image_value)
            if not image_url:
                logger.warning("No image at key '%s' in task %s", self.image_value, task.get("id"))
                predictions.append({"result": [], "score": 0.0})
                continue

            try:
                local_path = self.get_local_path(image_url)
                image = self._load_image(local_path)
                results = self._predict_single(image)
                avg_score = float(np.mean([r["score"] for r in results])) if results else 0.0
                logger.info("Task %s → %d predictions (avg score %.3f)", task.get("id"), len(results), avg_score)
                predictions.append({"result": results, "score": avg_score})
            except Exception:
                logger.error("Prediction failed for task %s", task.get("id"), exc_info=True)
                predictions.append({"result": [], "score": 0.0})

        return predictions

    def fit(self, annotations, **kwargs):
        """Acknowledge annotation feedback (no active retraining)."""
        logger.info("Received %d annotations for feedback", len(annotations))
        return {"status": "ok", "annotations_received": len(annotations)}

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _predict_single(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]

        # Step 1 — YOLO detection
        yolo_out = self.yolo(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        if not yolo_out.boxes or len(yolo_out.boxes) == 0:
            return []

        bboxes = yolo_out.boxes.xyxy.cpu().numpy()       # [N, 4] absolute pixels
        confs = yolo_out.boxes.conf.cpu().numpy()         # [N]
        cls_ids = yolo_out.boxes.cls.cpu().numpy().astype(int)
        class_names = yolo_out.names
        logger.info("YOLO: %d objects in %dx%d image", len(bboxes), w, h)

        # Step 2 — SAM2 segmentation using YOLO boxes as prompts
        masks = None
        try:
            sam_out = self.sam(image, bboxes=bboxes, verbose=False)[0]
            if sam_out.masks is not None:
                masks = sam_out.masks.data.cpu().numpy()   # [N, H, W]
                logger.info("SAM2: %d masks produced", len(masks))
        except Exception:
            logger.warning("SAM2 failed, using raw YOLO bboxes", exc_info=True)

        # Step 3 — Refine bboxes with SAM2 masks
        if masks is not None and len(masks) > 0:
            use_centroid = (self.refinement_strategy == "centroid")
            bboxes = refine_bboxes_batch(bboxes, masks, use_centroid=use_centroid)
            # Clip to image bounds
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)
        else:
            logger.info("No masks — using raw YOLO bboxes")

        # Step 4 — Convert to Label Studio format
        results = []
        for i, (bbox, conf, cls_id) in enumerate(zip(bboxes, confs, cls_ids)):
            x1, y1, x2, y2 = bbox
            results.append({
                "id": f"box_{i}",
                "from_name": self.bbox_from_name,
                "to_name": self.image_to_name,
                "type": "rectanglelabels",
                "score": float(conf),
                "original_width": int(w),
                "original_height": int(h),
                "image_rotation": 0,
                "value": {
                    "x": float(x1) / w * 100.0,
                    "y": float(y1) / h * 100.0,
                    "width": float(x2 - x1) / w * 100.0,
                    "height": float(y2 - y1) / h * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [class_names[cls_id]],
                },
            })

        return results

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))
