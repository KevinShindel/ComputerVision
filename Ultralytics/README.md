# Ultralytics


## Modes

| Name      | Description        | Documentation                                                                        |
|-----------|--------------------|--------------------------------------------------------------------------------------|
| val       | Model Validation   | [Params](https://docs.ultralytics.com/modes/val#arguments-for-yolo-model-validation) |
| train     | Model Training     | [Params](https://docs.ultralytics.com/modes/train#augmentation-settings-and-hyperparameters) |
| benchmark | Model Benchmarking | [Params](https://docs.ultralytics.com/modes/benchmark#arguments) |
| track     | Object Tracking    | [Params](https://docs.ultralytics.com/modes/track#shared-tracker-arguments) |
| predict   | Model Prediction   | [Params](https://docs.ultralytics.com/modes/predict#inference-sources)|
| export    | Model Export       | [Params](https://docs.ultralytics.com/modes/export#arguments) |


## Models

![models](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)


| Model        | Filenames                                                                                          | Task                  |
|--------------|----------------------------------------------------------------------------------------------------|-----------------------|
| YOLO26       | `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`                               | Detection             |
| YOLO26-seg   | `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`           | Instance Segmentation |
| YOLO26-sem   | `yolo26n-sem.pt`, `yolo26s-sem.pt`, `yolo26m-sem.pt`, `yolo26l-sem.pt`, `yolo26x-sem.pt`           | Semantic Segmentation |
| YOLO26-depth | `yolo26n-depth.pt`, `yolo26s-depth.pt`, `yolo26m-depth.pt`, `yolo26l-depth.pt`, `yolo26x-depth.pt` | Depth Estimation      |
| YOLO26-pose  | `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`, `yolo26l-pose.pt`, `yolo26x-pose.pt`      | Pose/Keypoints        |
| YOLO26-obb   | `yolo26n-obb.pt`, `yolo26s-obb.pt`, `yolo26m-obb.pt`, `yolo26l-obb.pt`, `yolo26x-obb.pt`           | Oriented Detection    |
| YOLO26-cls   | `yolo26n-cls.pt`, `yolo26s-cls.pt`, `yolo26m-cls.pt`, `yolo26l-cls.pt`, `yolo26x-cls.pt`           | Classification        | 

## Label Studio Integration

1. Install Label Studio: `pip install label-studio`
2. Start Label Studio: `label-studio start`
3. Create a new project and upload your dataset.