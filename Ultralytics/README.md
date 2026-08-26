# Ultralytics

## 📚 Ultralytics Learning Path

Ordered by folder name for a step-by-step flow:

1. **01-02** — 🔍 OpenCV basics and core image operations.
2. **01-04** — 🚀 Intro to Ultralytics usage and workflow.
3. **02-01** — 🏷️ Dataset labeling fundamentals using [label-studio](https://labelstud.io/).
4. **02-02** — ✂️ Dataset splitting and `data.yaml` setup.
5. **03-01** — 🧠 Ultralytics tasks and operating modes.
6. **03-02** — 🏋️ Detection model training with config + script.
7. **03-03** — 🤖 Auto-annotation workflow and trained weights using [Ultralytics Auto-Annotate](https://github.com/ultralytics/ultralytics).
8. **03-04** — 🧩 Segmentation model training.
9. **03-05** — 🕺 Pose/keypoint inference.
10. **03-06** — ✅ Model validation and metrics review.
11. **03-07** — 📦 Supported Ultralytics models overview.
12. **03-08** — 🎯 Tracking and prediction on media.
13. **03-09** — ⚡ Model benchmarking and performance checks.
14. **03-10** — 📤 Exporting models for deployment.
15. **04-01** — 🛠️ End-to-end Ultralytics solution notebook.
16. **04-02** — 🔢 Object counting solution.
17. **04-03** — 🗺️ Zone-based tracking solution.
18. **04-04** — 📊 Analytics-focused solution.
19. **04-05** — 💪 Workout monitoring with pose signals.
20. **04-06** — 🌐 Streamlit inference app and notebook.
21. **04-07** — 🧪 Final Ultralytics solution practice.

## Modes

| Name      | Description        | Documentation                                                                                |
|-----------|--------------------|----------------------------------------------------------------------------------------------|
| val       | Model Validation   | [Params](https://docs.ultralytics.com/modes/val#arguments-for-yolo-model-validation)         |
| train     | Model Training     | [Params](https://docs.ultralytics.com/modes/train#augmentation-settings-and-hyperparameters) |
| benchmark | Model Benchmarking | [Params](https://docs.ultralytics.com/modes/benchmark#arguments)                             |
| track     | Object Tracking    | [Params](https://docs.ultralytics.com/modes/track#shared-tracker-arguments)                  |
| predict   | Model Prediction   | [Params](https://docs.ultralytics.com/modes/predict#inference-sources)                       |
| export    | Model Export       | [Params](https://docs.ultralytics.com/modes/export#arguments)                                |


## Models

![models](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)


| Model        | Filenames                                                                                          | Task                  |
|--------------|----------------------------------------------------------------------------------------------------|-----------------------|
| YOLO26       | `yolo26n.pt.default`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`                               | Detection             |
| YOLO26-seg   | `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`           | Instance Segmentation |
| YOLO26-sem   | `yolo26n-sem.pt`, `yolo26s-sem.pt`, `yolo26m-sem.pt`, `yolo26l-sem.pt`, `yolo26x-sem.pt`           | Semantic Segmentation |
| YOLO26-depth | `yolo26n-depth.pt`, `yolo26s-depth.pt`, `yolo26m-depth.pt`, `yolo26l-depth.pt`, `yolo26x-depth.pt` | Depth Estimation      |
| YOLO26-pose  | `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`, `yolo26l-pose.pt`, `yolo26x-pose.pt`      | Pose/Keypoints        |
| YOLO26-obb   | `yolo26n-obb.pt`, `yolo26s-obb.pt`, `yolo26m-obb.pt`, `yolo26l-obb.pt`, `yolo26x-obb.pt`           | Oriented Detection    |
| YOLO26-cls   | `yolo26n-cls.pt`, `yolo26s-cls.pt`, `yolo26m-cls.pt`, `yolo26l-cls.pt`, `yolo26x-cls.pt`           | Classification        | 

## Label Studio Integration

1. Install Label Studio: `pip install label-studio`
2. Set max images `set DATA_UPLOAD_MAX_NUMBER_FILES=5000`
3. Start Label Studio: `label-studio start`
4. Create a new project and upload your dataset.