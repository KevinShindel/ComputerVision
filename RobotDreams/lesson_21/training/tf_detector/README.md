# TensorFlow Coin Detector

This folder contains a TensorFlow Object Detection API workflow for the euro coin dataset already stored in YOLO format under:

- `course_work/data/coin_yolo_split/images/{train,val,test}`
- `course_work/data/coin_yolo_split/labels/{train,val,test}`

## Files

- `prepare_tfrecord.py` — converts YOLO labels to TFRecord
- `label_map.pbtxt` — TensorFlow class mapping
- `infer_tf_detector.py` — runs exported TensorFlow detector
- `infer_tf_detector_with_cnn.py` — detector for bbox + existing Keras CNN for classification

## Recommended first detector

Start with **EfficientDet-D0**.

Good alternatives:
- SSD MobileNet: faster, lower accuracy
- Faster R-CNN: slower, often stronger on still images

## Steps

1. Create TFRecords:
   - `train.record`
   - `val.record`
   - optionally `test.record`

2. Download a TensorFlow Object Detection API pretrained model config/checkpoint:
   - `efficientdet_d0_coco17_tpu-32`
   - or `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`
   - or `faster_rcnn_resnet50_v1_640x640_coco17_tpu-8`

3. Edit `pipeline.config`:
   - set `num_classes: 8`
   - point `label_map_path` to `label_map.pbtxt`
   - point `train.record` and `val.record`
   - point `fine_tune_checkpoint` to downloaded checkpoint
   - set `fine_tune_checkpoint_type: "detection"`

4. Train with `model_main_tf2.py`

5. Export with `exporter_main_v2.py`

6. Run `infer_tf_detector.py`

## Notes

- TensorFlow detector outputs boxes in normalized `[ymin, xmin, ymax, xmax]`
- `infer_tf_detector.py` converts them to pixel `[x1, y1, x2, y2]`
- `infer_tf_detector_with_cnn.py` lets you keep your current classifier model and use detector only for localization