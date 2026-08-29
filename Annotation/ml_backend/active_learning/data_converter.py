"""Convert Label Studio annotations to YOLO format."""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class AnnotationToYoloConverter:
    """Converts Label Studio bbox annotations to YOLO format."""

    def __init__(
        self, image_width: int | None = None, image_height: int | None = None
    ):
        """
        Initialize converter.

        Args:
            image_width: Image width (auto-detected from image if not provided)
            image_height: Image height (auto-detected from image if not provided)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.class_mapping = {}  # label_name -> class_id

    def _normalize_bbox(
        self, x: float, y: float, w: float, h: float, img_w: int, img_h: int
    ) -> tuple:
        """Normalize bbox to YOLO format (center_x, center_y, width, height in 0-1 range)."""
        center_x = (x + w / 2) / img_w
        center_y = (y + h / 2) / img_h
        width_norm = w / img_w
        height_norm = h / img_h
        return center_x, center_y, width_norm, height_norm

    def _get_or_create_class_id(self, label_name: str) -> int:
        """Get or create a class ID for a label."""
        if label_name not in self.class_mapping:
            self.class_mapping[label_name] = len(self.class_mapping)
        return self.class_mapping[label_name]

    def process_task(
        self,
        task: dict,
        image_path: Path | None = None,
        bbox_tag_name: str = "bbox_labels",
        image_value_key: str = "image",
    ) -> tuple | None:
        """
        Process a single Label Studio task and convert to YOLO format.

        Args:
            task: Label Studio task dict with annotations
            image_path: Path to image file (for auto-detecting dimensions)
            bbox_tag_name: Name of bbox tag in Label Studio config
            image_value_key: Key for image path in task data

        Returns:
            (image_path, yolo_annotations) or None if no valid annotations
        """
        if "annotations" not in task or not task["annotations"]:
            return None

        # Use provided dimensions or auto-detect
        img_w = self.image_width
        img_h = self.image_height

        if img_w is None or img_h is None:
            if image_path is None:
                logger.warning(
                    f"Task {task.get('id')}: cannot determine image dimensions"
                )
                return None
            try:
                img = Image.open(image_path)
                img_w, img_h = img.size
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                return None

        yolo_annotations = []

        for annotation in task["annotations"]:
            if "result" not in annotation:
                continue

            for result in annotation["result"]:
                # Check if it's a rectangle label (bbox)
                if result.get("type") != "rectanglelabels":
                    continue

                value = result.get("value", {})
                if not value:
                    continue

                # Extract bbox coordinates (in pixels, 0-100 scale in Label Studio)
                x = value.get("x", 0)  # % of image width
                y = value.get("y", 0)  # % of image height
                width = value.get("width", 0)  # % of image width
                height = value.get("height", 0)  # % of image height

                # Convert from percentage to pixels
                x_px = (x / 100.0) * img_w
                y_px = (y / 100.0) * img_h
                w_px = (width / 100.0) * img_w
                h_px = (height / 100.0) * img_h

                # Get label
                labels = value.get("rectanglelabels", [])
                if not labels:
                    continue

                label_name = labels[0]
                class_id = self._get_or_create_class_id(label_name)

                # Normalize to YOLO format
                center_x, center_y, width_norm, height_norm = self._normalize_bbox(
                    x_px, y_px, w_px, h_px, img_w, img_h
                )

                yolo_annotations.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}"
                )

        if not yolo_annotations:
            return None

        return (image_path, yolo_annotations)

    def save_yolo_annotations(
        self, output_dir: Path, image_path: Path, annotations: list
    ) -> Path:
        """Save annotations in YOLO format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create txt file with same name as image
        txt_path = output_dir / f"{image_path.stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(annotations))

        return txt_path

    def get_class_mapping(self) -> dict:
        """Return the class name -> class_id mapping."""
        return self.class_mapping.copy()

    def save_classes_file(self, output_dir: Path) -> Path:
        """Save classes.txt file with label names (one per line)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create reverse mapping (class_id -> label_name)
        reverse_mapping = {v: k for k, v in self.class_mapping.items()}

        classes_path = output_dir / "classes.txt"
        with open(classes_path, "w") as f:
            f.writelines(f"{reverse_mapping[class_id]}\n" for class_id in sorted(reverse_mapping.keys()))

        logger.info(f"Saved classes file: {classes_path}")
        return classes_path
