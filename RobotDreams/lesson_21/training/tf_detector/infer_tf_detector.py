from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


TF_ID_TO_NAME = {
    1: "1 cent",
    2: "2 cent",
    3: "5 cent",
    4: "10 cent",
    5: "20 cent",
    6: "50 cent",
    7: "1 euro",
    8: "2 euro",
}


def load_detector(saved_model_dir: Path):
    return tf.saved_model.load(str(saved_model_dir))


def detect_objects(image_bgr, detector, conf_threshold=0.30):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb[None, ...], dtype=tf.uint8)

    outputs = detector(input_tensor)

    boxes = outputs["detection_boxes"][0].numpy()        # [ymin, xmin, ymax, xmax]
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(int)

    h, w = image_bgr.shape[:2]
    detections = []

    for box, score, cls_id in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue

        ymin, xmin, ymax, xmax = box

        x1 = int(round(xmin * w))
        y1 = int(round(ymin * h))
        x2 = int(round(xmax * w))
        y2 = int(round(ymax * h))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(cls_id),
            "class_name": TF_ID_TO_NAME.get(int(cls_id), str(cls_id)),
        })

    return detections


def draw_detections(image_bgr, detections):
    out = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["score"]:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def main():
    repo_root = Path(r"C:\Users\username\Projects\Computer-Vision-v2")
    image_path = repo_root / "course_work" / "data" / "test" / "euro_coins_example.jpg"

    # Example output of exporter_main_v2.py:
    # ...\course_work\models\tf_detector\efficientdet_d0\exported\saved_model
    saved_model_dir = repo_root / "course_work" / "models" / "tf_detector" / "efficientdet_d0" / "exported" / "saved_model"

    detector = load_detector(saved_model_dir)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = detect_objects(image_bgr, detector, conf_threshold=0.30)
    print(detections)

    annotated = draw_detections(image_bgr, detections)

    cv2.namedWindow("TensorFlow Detector", cv2.WINDOW_NORMAL)
    cv2.imshow("TensorFlow Detector", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()