# Active Learning Setup Guide

This guide explains how to enable incremental YOLO model training during manual annotation in LabelStudio.

## Overview

**Active Learning** enables your YOLO model to improve incrementally as you annotate:

```
Manual Annotation (in LabelStudio)
         ↓
    Submit annotations
         ↓
    /train endpoint called
         ↓
    Convert annotations to YOLO format
         ↓
    Fine-tune YOLO (few epochs)
         ↓
    Save checkpoint
         ↓
    Auto-reload in predictions
         ↓
    Next images get better predictions! 🎯
```

## Quick Start

### Step 1: Enable Training in LabelStudio

1. Open your LabelStudio **Project**
2. Go to **Settings** → **Model**
3. Under **Configuration**, check:
   - ✅ **"Start model training on annotation submission"**
4. Save

### Step 2: Configure .env

Update your `.env` file with Active Learning settings:

```ini
# Enable Active Learning
ACTIVE_LEARNING_ENABLED=true

# Training parameters (incremental = few epochs, fast)
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=640
AL_DEVICE=cpu

# Where to store training data
AL_TRAINING_DIR=./al-training
```

### Step 3: Run Backend

```bash
cd ml_backend
python server.py --host 0.0.0.0 --port 9090
```

### Step 4: Connect Model in LabelStudio

1. In LabelStudio Project, **Model** → **Connect**
2. Enter ML Backend URL: `http://localhost:9090`
3. Select your model
4. Click **Connect**

### Step 5: Start Annotating!

Just annotate normally. After each submission:
- Backend receives annotations
- Starts training in background (non-blocking)
- Saves improved checkpoint
- Next predictions use new model automatically

## Configuration Options

### .env Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_LEARNING_ENABLED` | `true` | Enable/disable training |
| `AL_EPOCHS` | `3` | Epochs per training session |
| `AL_BATCH_SIZE` | `4` | Batch size (lower = faster) |
| `AL_IMGSZ` | `640` | Image resolution for training |
| `AL_DEVICE` | `cpu` | Training device (`cpu`, `cuda`, `mps`) |
| `AL_TRAINING_DIR` | `./al-training` | Training data directory |

### Performance Tuning

**For fast training** (recommended for annotation):
```ini
AL_EPOCHS=2
AL_BATCH_SIZE=2
AL_IMGSZ=416
AL_DEVICE=cpu
```

**For better accuracy** (slower):
```ini
AL_EPOCHS=5
AL_BATCH_SIZE=8
AL_IMGSZ=640
AL_DEVICE=cuda
```

**For GPU training** (if available):
```ini
AL_DEVICE=cuda
AL_BATCH_SIZE=8
```

## Monitoring Training

### Check Training Progress

Look in `al-training/` directory:

```bash
ls -la al-training/
├── images/          # Training images (copied from LabelStudio)
├── labels/          # YOLO format annotations
├── checkpoints/     # Trained models
└── data.yaml        # Auto-generated YOLO config
```

### View Training Metadata

Each checkpoint has a `.json` file with metadata:

```bash
cat al-training/checkpoints/yolo_al_ep3_20260820_123456.json
```

Example output:
```json
{
  "timestamp": "20260820_123456",
  "epochs": 3,
  "batch_size": 4,
  "num_images": 45,
  "num_classes": 5,
  "class_mapping": {
    "person": 0,
    "car": 1,
    "bicycle": 2,
    ...
  }
}
```

### View Logs

The backend logs will show training progress:

```
2026-08-20 21:05:45 [INFO] Starting Active Learning training | annotations=3
2026-08-20 21:05:46 [INFO] Added 3 training samples
2026-08-20 21:06:15 [INFO] ✓ Training completed | checkpoint=al-training/checkpoints/yolo_al_ep3_20260820_210615.pt
2026-08-20 21:06:15 [INFO] Training stats: {'num_samples': 48, 'num_checkpoints': 5, ...}
```

## File Structure

```
Annotation/
├── ml_backend/
│   ├── active_learning/          # ← NEW
│   │   ├── __init__.py
│   │   ├── data_converter.py     # Convert annotations to YOLO format
│   │   ├── trainer.py            # Incremental training logic
│   │   └── README.md
│   ├── model.py                  # UPDATED: now includes train() method
│   ├── server.py
│   ├── .env                      # UPDATED: add AL settings
│   ├── .env.example              # UPDATED: document AL settings
│   └── ...
│
└── al-training/                  # ← NEW (created automatically)
    ├── images/                   # Training images
    ├── labels/                   # Annotations in YOLO format
    ├── checkpoints/              # Trained models
    │   ├── yolo_al_ep3_20260820_210615.pt
    │   ├── yolo_al_ep3_20260820_210615.json
    │   ├── yolo_al_ep3_20260820_210745.pt
    │   └── yolo_al_ep3_20260820_210745.json
    └── data.yaml                 # YOLO dataset config
```

## How It Works

### Data Flow

1. **User annotates in LabelStudio**
   - Creates bounding boxes with class labels
   - Submits annotation

2. **LabelStudio calls /train endpoint**
   - Sends all annotations in the request body
   - Backend acknowledges immediately

3. **Backend processes async**
   - Extracts annotations from request
   - Downloads/copies images
   - Converts annotations to YOLO format (normalized bbox coordinates)
   - Creates training dataset

4. **YOLO fine-tuning**
   - Loads base YOLO model
   - Trains for N epochs (default: 3)
   - Saves checkpoint with timestamp

5. **Auto-reload**
   - Next predict() call detects new checkpoint
   - Loads new model automatically
   - Future predictions use improved model

### Key Classes

#### AnnotationToYoloConverter
Converts Label Studio bbox annotations to YOLO format:
- Reads `rectanglelabels` from annotations
- Normalizes coordinates to 0-1 range
- Maintains class mapping
- Generates `.txt` files per image

#### ActiveLearningTrainer
Manages incremental training:
- Accumulates training samples
- Converts to YOLO dataset format
- Performs fine-tuning
- Saves checkpoints with metadata
- Loads latest checkpoint automatically

## Troubleshooting

### Model not updating?

**Check 1**: Is LabelStudio calling `/train`?
```bash
# Look for "Received N annotations" in backend logs
```

**Check 2**: Are annotations valid?
```bash
ls al-training/labels/
# Should see .txt files after annotations
```

**Check 3**: Is training enabled?
```bash
grep ACTIVE_LEARNING_ENABLED .env
# Should be "true"
```

### Out of memory?

Reduce training size:
```ini
AL_BATCH_SIZE=2          # From 4 to 2
AL_IMGSZ=416             # From 640 to 416
AL_EPOCHS=1              # From 3 to 1
```

### Training too slow?

Use CPU if GPU is overwhelmed:
```ini
AL_DEVICE=cpu            # Run on CPU instead
AL_BATCH_SIZE=2          # Smaller batch
```

Or reduce training:
```ini
AL_EPOCHS=1              # Just 1 epoch per submission
AL_IMGSZ=416             # Smaller images
```

### No images in al-training/images/?

The backend needs access to images. Check:
- Images are uploaded to LabelStudio
- `LABEL_STUDIO_URL` is correct in `.env`
- `LABEL_STUDIO_ACCESS_TOKEN` is set (if needed)

### Training fails with "No training labels available"?

Ensure your annotations have:
- Bounding boxes (rectanglelabels)
- Class labels assigned
- Boxes within image bounds

## Advanced Usage

### Disable Active Learning temporarily

```ini
ACTIVE_LEARNING_ENABLED=false
```

Backend will still acknowledge annotations but won't train.

### Use GPU for training

```ini
AL_DEVICE=cuda              # Requires CUDA-enabled GPU
AL_BATCH_SIZE=16            # Can be larger with GPU
```

### Clear training history

```bash
rm -rf al-training/
```

Next training will start fresh. Latest checkpoint will still be used for predictions if it's loaded.

### Monitor training in real-time

```bash
watch -n 1 'ls -lh al-training/checkpoints/'
```

Shows new checkpoints as they're saved.

### Inspect a trained model

```python
from ultralytics import YOLO

model = YOLO("al-training/checkpoints/yolo_al_ep3_20260820_210615.pt")
model.info()  # Print model info
```

## Best Practices

1. **Annotate consistently**
   - Ensure all objects of interest are labeled
   - Use consistent class names
   - Keep boxes tight around objects

2. **Start with enough data**
   - At least 20-30 samples before expecting improvement
   - More data = better incremental training

3. **Monitor model quality**
   - Check new predictions regularly
   - Adjust settings if predictions get worse
   - Save good checkpoints manually if needed

4. **Balance training frequency**
   - Training after EVERY annotation: fast iteration, potentially noisy
   - Training after batches: fewer updates, potentially better quality
   - Find what works for your workflow

5. **Resource management**
   - Use smaller images (416 instead of 640) on resource-limited systems
   - Reduce epochs if training takes too long
   - Consider using GPU if available

## FAQ

**Q: Will training block annotation?**
A: No! Training runs asynchronously in background. You can keep annotating.

**Q: Can I use GPU?**
A: Yes! Set `AL_DEVICE=cuda` if you have NVIDIA GPU.

**Q: What if I don't want to train on every annotation?**
A: Disable the LabelStudio setting. Annotation will work, but `/train` won't be called.

**Q: Can I manually trigger training?**
A: Yes! LabelStudio's model menu allows manual training triggers.

**Q: Are old checkpoints kept?**
A: Yes! All timestamped checkpoints are saved in `al-training/checkpoints/`.

**Q: How do I use a specific checkpoint?**
A: Manually set `YOLO_MODEL` to point to it in `.env`.

**Q: Does training improve predictions?**
A: Usually yes, if you have diverse, high-quality annotations. Quality matters more than quantity for incremental learning.

