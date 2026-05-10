from pathlib import Path
import hashlib
import json
import tensorflow as tf

# Reuse the same 8 classes already used in:
# - `course_work/data/coin_yolo_split/data.yaml`
# - `course_work/src/CNNStreamDetector.py`
CLASS_NAMES = [
    "1 cent",
    "2 cent",
    "5 cent",
    "10 cent",
    "20 cent",
    "50 cent",
    "1 euro",
    "2 euro",
]

YOLO_ID_TO_TF_ID = {i: i + 1 for i in range(len(CLASS_NAMES))}
YOLO_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}


def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(values):
    encoded = []
    for v in values:
        if isinstance(v, str):
            v = v.encode("utf-8")
        encoded.append(v)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in values]))


def float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in values]))


def find_images(images_dir: Path):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(images_dir.glob(pattern))
    return sorted(image_paths)


def decode_image_shape(image_bytes: bytes):
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.uint8)
    h = int(image.shape[0])
    w = int(image.shape[1])
    return h, w


def read_yolo_label_file(label_path: Path):
    if not label_path.exists():
        return []

    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    annotations = []
    for line in raw.splitlines():
        cls_id, x_center, y_center, width, height = map(float, line.split())
        annotations.append({
            "cls_id": int(cls_id),
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })
    return annotations


def yolo_to_normalized_xyxy(ann):
    cx = ann["x_center"]
    cy = ann["y_center"]
    bw = ann["width"]
    bh = ann["height"]

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return x1, y1, x2, y2


def create_tf_example(image_path: Path, label_path: Path):
    image_bytes = image_path.read_bytes()
    image_height, image_width = decode_image_shape(image_bytes)

    file_ext = image_path.suffix.lower().replace(".", "")
    image_format = "jpeg" if file_ext in ("jpg", "jpeg") else file_ext
    filename = image_path.name
    sha256 = hashlib.sha256(image_bytes).hexdigest()

    anns = read_yolo_label_file(label_path)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    class_texts, class_labels = [], []

    for ann in anns:
        x1, y1, x2, y2 = yolo_to_normalized_xyxy(ann)
        cls_id = ann["cls_id"]

        xmins.append(x1)
        xmaxs.append(x2)
        ymins.append(y1)
        ymaxs.append(y2)

        class_texts.append(YOLO_ID_TO_NAME[cls_id])
        class_labels.append(YOLO_ID_TO_TF_ID[cls_id])

    features = {
        "image/height": int64_feature(image_height),
        "image/width": int64_feature(image_width),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/key/sha256": bytes_feature(sha256),
        "image/encoded": bytes_feature(image_bytes),
        "image/format": bytes_feature(image_format),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(class_texts),
        "image/object/class/label": int64_list_feature(class_labels),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(images_dir: Path, labels_dir: Path, output_path: Path):
    image_paths = find_images(images_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_examples = 0
    num_missing_labels = 0

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                num_missing_labels += 1

            example = create_tf_example(image_path, label_path)
            writer.write(example.SerializeToString())
            num_examples += 1

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_path": str(output_path),
        "num_examples": num_examples,
        "num_missing_labels": num_missing_labels,
    }


def main():
    # Matches your repo layout around:
    # `course_work/data/coin_yolo_split/data.yaml`
    repo_root = Path(r"C:\Users\username\Projects\Computer-Vision-v2")
    data_root = repo_root / "course_work" / "data" / "coin_yolo_split"

    outputs = []

    for split in ("train", "val", "test"):
        images_dir = data_root / "images" / split
        labels_dir = data_root / "labels" / split

        if not images_dir.exists():
            continue

        output_path = data_root / f"{split}.record"
        stats = write_tfrecord(images_dir, labels_dir, output_path)
        outputs.append(stats)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()