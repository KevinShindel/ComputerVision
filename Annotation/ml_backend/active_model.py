"""
YOLO26 + SAM2 Label Studio ML Backend with Active Learning.

Extends YoloSamBackend with incremental YOLO fine-tuning capabilities.

Pipeline:
  1. YOLO26 detects objects → bounding boxes + class labels
  2. SAM2 uses those boxes as prompts → pixel-level masks per object
  3. Masks refine bounding box coordinates (corrects YOLO inaccuracies)
  4. Results are returned as RectangleLabels with refined coordinates
  5. NEW: On annotation events, fine-tunes YOLO on labeled data
"""

import logging
import os

from active_learning import ActiveLearningTrainer
from model import YoloSamBackend
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ActiveYoloSamBackend(YoloSamBackend):
    """YOLO26 + SAM2 backend with Active Learning for incremental model training."""

    # Include START_TRAINING so process_event() calls fit() for it too,
    # preventing the framework from writing an empty result file.
    TRAIN_EVENTS = YoloSamBackend.TRAIN_EVENTS + ("START_TRAINING",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── Active Learning setup ───────────────────────────────────────────────
        self.al_enabled = os.getenv("ACTIVE_LEARNING_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.al_trainer = None
        self.yolo_base_path = _resolve_model_path("YOLO_MODEL", "models/yolo26s.pt")

        if self.al_enabled:
            try:
                al_training_dir = os.getenv("AL_TRAINING_DIR", "./al-training")
                al_epochs = int(os.getenv("AL_EPOCHS", "3"))
                al_batch_size = int(os.getenv("AL_BATCH_SIZE", "4"))
                al_imgsz = int(os.getenv("AL_IMGSZ", "640"))
                al_device = os.getenv("AL_DEVICE", "cpu")

                self.al_trainer = ActiveLearningTrainer(
                    base_model_path=self.yolo_base_path,
                    training_dir=al_training_dir,
                    epochs=al_epochs,
                    batch_size=al_batch_size,
                    imgsz=al_imgsz,
                    device=al_device,
                )
                # Pre-populate class mapping from Label Studio label config so
                # class IDs are deterministic across all subprocess restarts.
                self._seed_class_mapping()
                logger.info(
                    "Active Learning enabled | epochs=%d | batch=%d | device=%s",
                    al_epochs,
                    al_batch_size,
                    al_device,
                )
            except Exception as e:
                logger.error("Failed to initialize Active Learning: %s", e)
                self.al_enabled = False
        else:
            logger.info("Active Learning disabled")

    def predict(self, tasks, **kwargs):
        """Override predict to auto-load latest trained checkpoint."""
        if self.al_enabled and self.al_trainer:
            latest_checkpoint = self.al_trainer.get_latest_checkpoint()
            if latest_checkpoint and str(latest_checkpoint) != self.yolo_base_path:
                try:
                    logger.info("Loading latest AL checkpoint: %s", latest_checkpoint)
                    self.yolo = YOLO(str(latest_checkpoint))
                    self.yolo_base_path = str(latest_checkpoint)
                except Exception as e:
                    logger.error("Failed to load checkpoint: %s", e)

        return super().predict(tasks, **kwargs)

    def fit(self, annotations, **kwargs):
        """Add the new annotation to the training set and fine-tune synchronously.

        The label_studio_ml framework always calls fit((), event=..., data=...) from
        inside a dedicated subprocess, so there is no need for a background thread —
        running synchronously here is both correct and safe.
        """
        if not self.al_enabled or not self.al_trainer:
            logger.info("Active Learning disabled — acknowledgment only")
            return {"status": "ok", "message": "Active Learning disabled"}

        event = kwargs.get("event", "")
        data = kwargs.get("data", {})

        # START_TRAINING carries only project metadata — retrain on whatever
        # samples have already been accumulated on disk.
        if event == "START_TRAINING":
            logger.info("START_TRAINING received — retraining on accumulated samples")
            return self._run_training()

        # For ANNOTATION_* events the payload contains both task and annotation.
        task = data.get("task")
        annotation = data.get("annotation")

        if not task or not annotation:
            logger.warning("Event %s: no task/annotation in payload — skipping", event)
            return {"status": "ok", "message": "No annotation data in event"}

        # Build the task structure expected by add_training_sample / process_task.
        task_for_training = dict(task)
        task_for_training["annotations"] = [annotation]
        raw_url = task.get("data", {}).get(self.image_value)
        image_url = self._resolve_image_url(raw_url)

        logger.info(
            "Adding sample | annotation=%s task=%s",
            annotation.get("id"),
            task.get("id"),
        )
        added = self.al_trainer.add_training_sample(task_for_training, image_url)
        if not added:
            logger.warning(
                "Sample not added for task %s — skipping training", task.get("id")
            )
            return {"status": "ok", "message": "Sample not added"}

        logger.info(
            "Sample added | total on disk=%d",
            self.al_trainer.get_num_training_samples(),
        )
        return self._run_training()

    def _run_training(self) -> dict:
        """Run YOLO fine-tuning and return a result dict suitable for the framework."""
        try:
            checkpoint = self.al_trainer.train()
        except Exception as e:
            logger.error("Training raised an exception: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}

        if checkpoint:
            stats = self.al_trainer.get_training_stats()
            logger.info(
                "Training completed | checkpoint=%s | stats=%s", checkpoint, stats
            )
            return {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "num_samples": stats.get("num_samples", 0),
            }

        logger.warning("Training produced no checkpoint (not enough data?)")
        return {"status": "ok", "message": "Training skipped — not enough data"}

    def _seed_class_mapping(self) -> None:
        """Extract label names from the Label Studio config and seed the trainer."""
        if not self.al_trainer:
            return
        try:
            # parsed_label_config: {tag_name: {'labels': [...], ...}, ...}
            for tag_info in self.parsed_label_config.values():
                labels = tag_info.get("labels", [])
                if labels:
                    self.al_trainer.seed_class_mapping(labels)
                    return
        except Exception as e:
            logger.warning("Could not seed class mapping from label config: %s", e)

    def _resolve_image_url(self, url: str) -> str:
        """Convert a Label Studio-internal URL (e.g. /data/upload/…) to a local path.

        LabelStudioMLBase.get_local_path() handles the hostname + token lookup,
        local-files mapping, and temporary download as needed.
        """
        if not url:
            return url
        try:
            return self.get_local_path(url)
        except Exception as e:
            logger.warning(
                "Could not resolve image URL %r: %s — using raw value", url, e
            )
            return url


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
