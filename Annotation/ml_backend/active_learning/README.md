"""README for Active Learning Module."""

# Active Learning Module for YOLO

This module enables **incremental fine-tuning** of YOLO models during annotation in Label Studio.

## Overview

When enabled, the `/train` endpoint will:
1. **Receive annotations** from Label Studio
2. **Convert them to YOLO format** (normalized bounding boxes)
3. **Fine-tune the YOLO model** with new data
4. **Auto-save checkpoints** with timestamps
5. **Automatically update predictions** to use the latest trained model

## Setup

### 1. Enable Training in LabelStudio

In your Project settings → Model → Configuration:
- ✅ Check: **"Start model training on annotation submission"**
- This tells Label Studio to call `/train` when you submit annotations

### 2. Update .env

Add these settings to `.env`:

```ini
# Active Learning
ACTIVE_LEARNING_ENABLED=true
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=640
AL_TRAINING_DIR=./al-training
```

| Setting | Default | Description |
|---------|---------|-------------|
| `ACTIVE_LEARNING_ENABLED` | `true` | Enable/disable training |
| `AL_EPOCHS` | `3` | Epochs per training session (few = fast incremental) |
| `AL_BATCH_SIZE` | `4` | Batch size (smaller = faster) |
| `AL_IMGSZ` | `640` | Training image size |
| `AL_TRAINING_DIR` | `./al-training` | Where to store training data & checkpoints |

### 3. Run the Backend

```bash
cd ml_backend
python server.py
```

Then in Label Studio, **Connect** the ML Backend model.

## Workflow

```
User annotates in Label Studio
              ↓
          Submit annotation
              ↓
Label Studio calls /train endpoint
              ↓
Active Learning Trainer processes annotations
              ↓
YOLO model fine-tunes for few epochs
              ↓
New checkpoint saved with timestamp
              ↓
Backend auto-loads latest model
              ↓
Next predictions use improved model!
```

## File Structure

```
ml_backend/active_learning/
├── __init__.py              # Module exports
├── data_converter.py        # Label Studio → YOLO format converter
├── trainer.py              # YOLO fine-tuning logic
└── README.md               # This file

al-training/               # Created automatically
├── images/                # Training images (copied from Label Studio)
├── labels/                # YOLO format annotations (*.txt)
├── checkpoints/           # Trained model checkpoints with timestamps
└── data.yaml              # Auto-generated YOLO dataset config
```

## Key Classes

### `AnnotationToYoloConverter`

Converts Label Studio bbox annotations to YOLO normalized format.

```python
converter = AnnotationToYoloConverter()
result = converter.process_task(task, image_path)
# Returns: (image_path, [yolo_annotations])
```

**YOLO format**: `class_id center_x center_y width height`
- Coordinates are normalized to 0-1 range
- One annotation per line in `.txt` file

### `ActiveLearningTrainer`

Manages incremental training workflow.

```python
trainer = ActiveLearningTrainer(
    base_model_path="models/yolo26s.pt",
    epochs=3,
    batch_size=4,
    device="cpu"
)

# Add samples from annotations
trainer.add_training_sample(task)

# Fine-tune
checkpoint = trainer.train()

# Get latest model
latest = trainer.get_latest_checkpoint()
```

## Integration with server.py

The `/train` endpoint is auto-implemented by Label Studio ML framework. You just need to add:

```python
def train(self, tasks, **kwargs):
    """
    Implement Label Studio /train endpoint for active learning.
    Called when 'Start model training on annotation submission' is enabled.
    """
    # See model.py for implementation
```

## Monitoring Training

Check the training directory:

```bash
ls -la al-training/
# images/      - training images
# labels/      - annotations
# checkpoints/ - trained models
# data.yaml    - dataset config
```

View a checkpoint's metadata:

```bash
cat al-training/checkpoints/yolo_al_ep3_20260820_123456.json
```

## Troubleshooting

### Model not updating?
- Check logs for errors in `/train` endpoint
- Verify annotations are valid (bboxes within image bounds)
- Ensure `ACTIVE_LEARNING_ENABLED=true` in `.env`

### Out of memory?
- Reduce `AL_BATCH_SIZE` (try 2 or 1)
- Reduce `AL_EPOCHS` (try 1)
- Set device: `AL_DEVICE=cpu`

### Training too slow?
- Use smaller `AL_IMGSZ` (try 416)
- Reduce `AL_BATCH_SIZE`
- Use GPU: set device to `cuda` or `mps`

## Performance Tuning

**For fast incremental learning** (recommended):
```ini
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=416
```

**For better accuracy** (slower):
```ini
AL_EPOCHS=5
AL_BATCH_SIZE=8
AL_IMGSZ=640
```

## Checkpoints

Each training session creates:
- `yolo_al_ep{epochs}_{timestamp}.pt` - Trained model weights
- `yolo_al_ep{epochs}_{timestamp}.json` - Metadata (class mapping, stats)

Latest checkpoint is auto-loaded for predictions.

## Disabling Active Learning

To disable training:
```ini
ACTIVE_LEARNING_ENABLED=false
```

Or simply don't enable the setting in Label Studio.
