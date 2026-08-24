"""
YOLO26 + SAM2 Label Studio ML Backend with Active Learning.

Extends YoloSamBackend with incremental YOLO fine-tuning capabilities.

Pipeline:
  1. YOLO26 detects objects → bounding boxes + class labels
  2. SAM2 uses those boxes as prompts → pixel-level masks per object
  3. Masks refine bounding box coordinates (corrects YOLO inaccuracies)
  4. Results are returned as RectangleLabels with refined coordinates
  5. NEW: On /train endpoint, fine-tunes YOLO on labeled data
"""

import logging
import os
from threading import Thread

from active_learning import ActiveLearningTrainer
from model import YoloSamBackend
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ActiveYoloSamBackend(YoloSamBackend):
    """YOLO26 + SAM2 backend with Active Learning for incremental model training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── Active Learning setup ───────────────────────────────────────────────
        self.al_enabled = os.getenv("ACTIVE_LEARNING_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.al_trainer = None
        self.al_training_thread = None
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
        # Check if there's a new trained checkpoint and reload
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
        """Process annotations and trigger Active Learning training."""
        logger.info("Received %d annotations", len(annotations))

        if not self.al_enabled or not self.al_trainer:
            logger.info("Active Learning disabled — acknowledgment only")
            return {"status": "ok", "annotations_received": len(annotations)}

        # Process annotations asynchronously
        self.al_training_thread = Thread(
            target=self._train_async,
            args=(annotations,),
            daemon=True,
        )
        self.al_training_thread.start()

        return {
            "status": "training_queued",
            "annotations_received": len(annotations),
            "message": "Training started in background",
        }

    def _train_async(self, annotations):
        """Async training worker thread."""
        try:
            logger.info(
                "Starting Active Learning training | annotations=%d", len(annotations)
            )

            # Aggregate tasks from annotations
            tasks_dict = {}
            for ann in annotations:
                task_id = ann.get("task")
                if task_id not in tasks_dict:
                    tasks_dict[task_id] = {
                        "id": task_id,
                        "data": ann.get("data", {}),
                        "annotations": [],
                    }
                tasks_dict[task_id]["annotations"].append(ann)

            # Add samples to trainer
            num_added = 0
            for task_id, task in tasks_dict.items():
                image_url = task.get("data", {}).get(self.image_value)
                if self.al_trainer.add_training_sample(task, image_url):
                    num_added += 1

            if num_added == 0:
                logger.warning("No valid training samples added")
                return

            logger.info("Added %d training samples", num_added)

            # Train
            checkpoint = self.al_trainer.train()
            if checkpoint:
                logger.info("✓ Training completed | checkpoint=%s", checkpoint)
                stats = self.al_trainer.get_training_stats()
                logger.info("Training stats: %s", stats)
            else:
                logger.error("Training failed")

        except Exception as e:
            logger.error("Async training failed: %s", e, exc_info=True)


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
