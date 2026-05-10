from pathlib import Path
from collections import Counter
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

CNN_ID_TO_NAME = {
    0: "1 Cent",
    1: "2 Cent",
    2: "5 Cent",
    3: "10 Cent",
    4: "20 Cent",
    5: "50 Cent",
    6: "1 Euro",
    7: "2 Euro",
}

COIN_TO_CENTS = {
    "1 Cent": 1,
    "2 Cent": 2,
    "5 Cent": 5,
    "10 Cent": 10,
    "20 Cent": 20,
    "50 Cent": 50,
    "1 Euro": 100,
    "2 Euro": 200,
}


def load_detector(saved_model_dir: Path):
    return tf.saved_model.load(str(saved_model_dir))


def load_classifier(model_path: Path):
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_crop_for_model(crop_bgr, image_size=(128, 128), scale01=False):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, image_size, interpolation=cv2.INTER_AREA)
    x = crop_rgb.astype("float32")
    if scale01:
        x = x / 255.0
    return x


def detect_objects(image_bgr, detector, conf_threshold=0.30):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb[None, ...], dtype=tf.uint8)

    outputs = detector(input_tensor)
    boxes = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()

    h, w = image_bgr.shape[:2]
    results = []

    for box, score in zip(boxes, scores):
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

        results.append({
            "bbox": [x1, y1, x2, y2],
            "detector_score": float(score),
        })

    return results


def classify_boxes(image_bgr, boxes, classifier, image_size=(128, 128), scale01=False):
    crops = []
    valid_items = []

    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        x_model = preprocess_crop_for_model(crop, image_size=image_size, scale01=scale01)
        crops.append(x_model)
        valid_items.append(item)

    if not crops:
        return [], Counter(), 0

    batch = np.stack(crops, axis=0)
    probs = classifier.predict(batch, verbose=0)

    predictions = []
    counts = Counter()
    total_cents = 0

    for item, p in zip(valid_items, probs):
        pred_idx = int(np.argmax(p))
        pred_name = CNN_ID_TO_NAME[pred_idx]
        pred_conf = float(p[pred_idx])

        output = {
            "bbox": item["bbox"],
            "detector_score": item["detector_score"],
            "class_id": pred_idx,
            "class_name": pred_name,
            "class_confidence": pred_conf,
        }
        predictions.append(output)

        counts[pred_name] += 1
        total_cents += COIN_TO_CENTS[pred_name]

    return predictions, counts, total_cents


def draw_predictions(image_bgr, predictions, total_cents):
    out = image_bgr.copy()

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        label = f'{pred["class_name"]} det={pred["detector_score"]:.2f} cls={pred["class_confidence"]:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    euros = total_cents // 100
    cents = total_cents % 100
    cv2.putText(
        out,
        f"Total: {euros} euro {cents} cent",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def main():
    repo_root = Path(r"C:\Users\username\Projects\Computer-Vision-v2")
    image_path = repo_root / "course_work" / "data" / "test" / "euro_coins_example.jpg"

    detector_path = repo_root / "course_work" / "models" / "tf_detector" / "efficientdet_d0" / "exported" / "saved_model"
    classifier_path = repo_root / "course_work" / "models" / "tensorflow" / "weights" / "best_model.keras"

    detector = load_detector(detector_path)
    classifier = load_classifier(classifier_path)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    boxes = detect_objects(image_bgr, detector, conf_threshold=0.30)
    predictions, counts, total_cents = classify_boxes(
        image_bgr,
        boxes,
        classifier,
        image_size=(128, 128),
        scale01=False,
    )

    print(predictions)
    print(dict(counts), total_cents)

    annotated = draw_predictions(image_bgr, predictions, total_cents)

    cv2.namedWindow("TF Detector + CNN Classifier", cv2.WINDOW_NORMAL)
    cv2.imshow("TF Detector + CNN Classifier", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()