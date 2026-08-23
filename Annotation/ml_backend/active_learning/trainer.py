"""Active Learning trainer for incremental YOLO fine-tuning."""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from ultralytics import YOLO

from .data_converter import AnnotationToYoloConverter

logger = logging.getLogger(__name__)


class ActiveLearningTrainer:
    """Handles incremental YOLO fine-tuning from Label Studio annotations."""

    def __init__(
        self,
        base_model_path: str,
        training_dir: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 4,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        """
        Initialize Active Learning trainer.

        Args:
            base_model_path: Path to base YOLO model
            training_dir: Directory for training data and checkpoints
            epochs: Number of epochs per training session
            batch_size: Batch size for training
            imgsz: Input image size
            device: Training device (cpu, cuda, etc.)
        """
        self.base_model_path = base_model_path
        self.training_dir = Path(training_dir or "./al-training")
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device

        # Directories for training data
        self.images_dir = self.training_dir / "images"
        self.labels_dir = self.training_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.checkpoint_dir = self.training_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_model = None
        self.converter = AnnotationToYoloConverter()

        logger.info(
            f"ActiveLearningTrainer initialized | model={base_model_path} | "
            f"epochs={epochs} | device={device}"
        )

    def _download_image(self, image_url: str) -> Optional[Path]:
        """Download image from Label Studio URL."""
        try:
            # If it's a local path, just return it
            if os.path.isfile(image_url):
                return Path(image_url)

            # Try to download from URL
            import requests

            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Save to temp file
            ext = Path(image_url).suffix or ".jpg"
            temp_path = self.images_dir / f"temp_{datetime.now().timestamp()}{ext}"
            with open(temp_path, "wb") as f:
                f.write(response.content)

            return temp_path
        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            return None

    def add_training_sample(self, task: dict, image_url: Optional[str] = None) -> bool:
        """
        Add a single task's annotations to training dataset.

        Args:
            task: Label Studio task dict
            image_url: Image URL (from task data if not provided)

        Returns:
            True if sample was added, False otherwise
        """
        if image_url is None:
            # Try to get from task data
            image_url = task.get("data", {}).get("image")

        if not image_url:
            logger.warning(f"Task {task.get('id')}: no image URL provided")
            return False

        # Download or locate image
        image_path = self._download_image(image_url)
        if not image_path or not image_path.exists():
            logger.warning(f"Could not access image: {image_url}")
            return False

        # Convert annotations
        result = self.converter.process_task(task, image_path)
        if result is None:
            logger.debug(f"Task {task.get('id')}: no valid annotations")
            return False

        img_path, yolo_annotations = result

        # Copy image to training directory with unique name
        task_id = task.get("id", "unknown")
        new_image_path = self.images_dir / f"task_{task_id}_{image_path.name}"
        try:
            shutil.copy2(img_path, new_image_path)
        except Exception as e:
            logger.error(f"Failed to copy image: {e}")
            return False

        # Save YOLO annotations
        try:
            txt_path = self.labels_dir / f"task_{task_id}_{image_path.stem}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))
            logger.info(f"Added training sample: {new_image_path.name} with {len(yolo_annotations)} objects")
            return True
        except Exception as e:
            logger.error(f"Failed to save annotations: {e}")
            return False

    def _create_data_yaml(self) -> Path:
        """Create data.yaml for YOLO training."""
        data_yaml_path = self.training_dir / "data.yaml"

        class_mapping = self.converter.get_class_mapping()
        class_names = {v: k for k, v in class_mapping.items()}

        # Sort by class_id
        sorted_classes = [class_names[i] for i in sorted(class_names.keys())]

        data_yaml_content = f"""path: {self.training_dir.absolute()}
train: images
val: images

nc: {len(sorted_classes)}
names: {sorted_classes}
"""

        with open(data_yaml_path, "w") as f:
            f.write(data_yaml_content)

        logger.info(f"Created data.yaml with {len(sorted_classes)} classes")
        return data_yaml_path

    def train(self) -> Optional[Path]:
        """
        Fine-tune YOLO model on accumulated training data.

        Returns:
            Path to trained model or None if training failed
        """
        # Check if we have training data
        if not list(self.images_dir.glob("*.jpg")) and not list(self.images_dir.glob("*.png")):
            logger.warning("No training images available")
            return None

        if len(list(self.labels_dir.glob("*.txt"))) == 0:
            logger.warning("No training labels available")
            return None

        try:
            # Create data.yaml
            data_yaml = self._create_data_yaml()

            # Load base model
            logger.info(f"Loading base model: {self.base_model_path}")
            model = YOLO(self.base_model_path)

            # Train
            logger.info(
                f"Starting training | epochs={self.epochs} | batch_size={self.batch_size} | "
                f"device={self.device}"
            )
            results = model.train(
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                verbose=True,
                patience=0,  # No early stopping for incremental training
            )

            # Save checkpoint with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"yolo_al_ep{self.epochs}_{timestamp}.pt"
            model.save(str(checkpoint_path))

            self.current_model = checkpoint_path

            logger.info(f"Training completed | checkpoint saved: {checkpoint_path}")

            # Save training metadata
            metadata = {
                "timestamp": timestamp,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_images": len(list(self.images_dir.glob("*"))),
                "num_classes": len(self.converter.get_class_mapping()),
                "class_mapping": self.converter.get_class_mapping(),
            }

            metadata_path = checkpoint_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return checkpoint_path

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent trained checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("yolo_al_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def get_num_training_samples(self) -> int:
        """Get number of training samples accumulated."""
        return len(list(self.images_dir.glob("*")))

    def clear_training_data(self) -> None:
        """Clear all accumulated training data."""
        for f in self.images_dir.glob("*"):
            f.unlink()
        for f in self.labels_dir.glob("*"):
            f.unlink()
        logger.info("Cleared training data")

    def get_training_stats(self) -> dict:
        """Get training dataset statistics."""
        return {
            "num_samples": self.get_num_training_samples(),
            "num_checkpoints": len(list(self.checkpoint_dir.glob("*.pt"))),
            "classes": self.converter.get_class_mapping(),
            "latest_checkpoint": str(self.get_latest_checkpoint()) if self.get_latest_checkpoint() else None,
        }
