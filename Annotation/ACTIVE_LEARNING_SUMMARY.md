# Active Learning Implementation Summary

## What Was Created

A complete **Active Learning** system for incremental YOLO fine-tuning during LabelStudio annotation.

### New Module: `ml_backend/active_learning/`

```
ml_backend/active_learning/
├── __init__.py                 # Module exports
├── data_converter.py           # Label Studio → YOLO format converter
├── trainer.py                  # Incremental YOLO fine-tuning logic  
└── README.md                   # Module documentation
```

### Modified Files

- **`ml_backend/model.py`**
  - Added Active Learning initialization
  - Implemented `train()` endpoint via `fit()` method
  - Auto-loads latest trained checkpoint in predictions
  - Runs training asynchronously (non-blocking)

- **`ml_backend/.env.example`**
  - Added Active Learning configuration variables

## Key Features

✅ **Incremental Fine-tuning**
- Trains YOLO on new annotations with few epochs
- Fast training (3 epochs by default, ~2-3 min on CPU)
- Configurable epochs, batch size, image resolution

✅ **Automatic Model Updates**
- Detects new checkpoints automatically
- Loads latest model for predictions
- No manual intervention needed

✅ **Non-Blocking Training**
- Runs in background thread
- Annotation submission returns immediately
- You can keep annotating while training

✅ **Training Data Management**
- Converts Label Studio annotations to YOLO format
- Organizes training data in `al-training/` directory
- Saves all checkpoints with timestamps and metadata

✅ **Easy Integration**
- Works with existing `server.py`
- Just add to `.env` and enable in LabelStudio
- Backward compatible (disable if not needed)

## Setup (Quick)

### 1. Update `.env`
```ini
ACTIVE_LEARNING_ENABLED=true
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=640
AL_DEVICE=cpu
AL_TRAINING_DIR=./al-training
```

### 2. Enable in LabelStudio
Settings → Model → Configuration → ✅ "Start model training on annotation submission"

### 3. Run backend
```bash
cd ml_backend
python server.py
```

### 4. Annotate & watch model improve!

## File Structure After First Training

```
Annotation/
├── ml_backend/
│   ├── active_learning/          ← NEW MODULE
│   ├── model.py                  ← UPDATED
│   ├── .env                       ← UPDATE THIS
│   ├── .env.example              ← UPDATED
│   └── ...
│
├── ACTIVE_LEARNING_SETUP.md       ← SETUP GUIDE (detailed)
└── ACTIVE_LEARNING_SUMMARY.md     ← THIS FILE
```

Plus new directory created on first training:
```
al-training/                       ← AUTO-CREATED
├── images/                        # Training images
├── labels/                        # Annotations (.txt files)
├── checkpoints/                   # Trained models
└── data.yaml                      # YOLO config
```

## How It Works

```
1. Annotate in LabelStudio
        ↓
2. Submit annotations
        ↓
3. LabelStudio calls /train endpoint
        ↓
4. Backend (model.py) receives annotations
        ↓
5. ActiveLearningTrainer processes async:
   a. AnnotationToYoloConverter converts to YOLO format
   b. Saves images & labels to al-training/
   c. Runs YOLO.train() for N epochs
   d. Saves checkpoint with timestamp
        ↓
6. Next prediction:
   - predict() detects new checkpoint
   - Loads improved model automatically
   - Returns better predictions
        ↓
7. Next annotations train on improved model
        ↓
8. Repeat → Model continuously improves! 🎯
```

## Configuration Options

Add to `.env`:

| Setting | Default | Tuning |
|---------|---------|--------|
| `ACTIVE_LEARNING_ENABLED` | `true` | Set to `false` to disable |
| `AL_EPOCHS` | `3` | Lower for speed, higher for quality |
| `AL_BATCH_SIZE` | `4` | Lower if out of memory |
| `AL_IMGSZ` | `640` | Lower (416) for speed, higher for quality |
| `AL_DEVICE` | `cpu` | Use `cuda` if GPU available |
| `AL_TRAINING_DIR` | `./al-training` | Where to store training data |

## Performance Characteristics

**Fast incremental learning** (recommended):
- 3 epochs, batch=4, image_size=416, CPU
- ~2-3 minutes per training session

**Better quality** (slower):
- 5 epochs, batch=8, image_size=640, GPU
- ~10-15 minutes per training session

**Very fast** (minimal improvement):
- 1 epoch, batch=2, image_size=416, CPU
- ~30-60 seconds per training session

## Monitoring

### Check Training Progress
```bash
# See training data accumulating
ls -la al-training/images/
ls -la al-training/labels/

# Watch checkpoints being saved
watch ls -lh al-training/checkpoints/
```

### View Logs
Backend logs will show:
```
[INFO] Received 3 annotations
[INFO] Added 3 training samples
[INFO] Starting Active Learning training
[INFO] ✓ Training completed | checkpoint=al-training/checkpoints/yolo_al_ep3_20260820_210615.pt
[INFO] Training stats: {'num_samples': 45, 'num_checkpoints': 5, ...}
```

### Check Checkpoint Metadata
```bash
cat al-training/checkpoints/yolo_al_ep3_*.json
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not updating | Check `/train` is called (see logs). Verify annotations have bboxes. |
| Out of memory | Reduce `AL_BATCH_SIZE` (2 or 1), reduce `AL_IMGSZ` (416) |
| Training too slow | Use `AL_DEVICE=cpu` for speed. Reduce epochs. |
| No images saved | Check images are uploaded to LabelStudio and backend can access them. |

See **ACTIVE_LEARNING_SETUP.md** for detailed troubleshooting.

## Code Overview

### `active_learning/data_converter.py`

```python
converter = AnnotationToYoloConverter()
result = converter.process_task(task, image_path)
# Returns: (image_path, ["0 0.5 0.5 0.2 0.3", ...])  # YOLO format
```

Converts Label Studio bbox annotations to YOLO normalized format:
- Reads `rectanglelabels` from annotations
- Normalizes coordinates to 0-1 range
- Maintains class ID mapping
- Generates `.txt` files per image

### `active_learning/trainer.py`

```python
trainer = ActiveLearningTrainer(
    base_model_path="models/yolo26s.pt",
    epochs=3,
    batch_size=4,
)

trainer.add_training_sample(task)
checkpoint = trainer.train()
latest = trainer.get_latest_checkpoint()
```

Manages incremental training pipeline:
- Accumulates training samples
- Converts to YOLO dataset format
- Performs fine-tuning
- Saves checkpoints with metadata
- Provides training statistics

### `model.py` Changes

- `__init__`: Initialize `ActiveLearningTrainer` if enabled
- `predict()`: Load latest checkpoint if available
- `fit()`: Process annotations and trigger async training
- `_train_async()`: Worker thread for non-blocking training

## Restrictions Honored

✅ **No existing notebooks modified** - all code is in new `active_learning/` folder  
✅ **New folder structure** - separate from original setup  
✅ **Backward compatible** - can be disabled with one config flag  

## Next Steps

1. **Update `.env`** - Add Active Learning settings (copy from `.env.example`)
2. **Enable in LabelStudio** - Project Settings → Model → Configuration checkbox
3. **Start backend** - `python server.py`
4. **Connect model** - LabelStudio → Model → Connect
5. **Annotate** - Watch model improve with each annotation!

## Documentation

- **`ACTIVE_LEARNING_SETUP.md`** - Complete setup and troubleshooting guide
- **`active_learning/README.md`** - Technical module documentation
- **`ml_backend/.env.example`** - Configuration reference

## Benefits

- 🚀 **Fast iteration** - Incremental training during annotation
- 🎯 **Better predictions** - Model learns from your annotations
- ⚡ **Non-blocking** - Training doesn't interrupt workflow
- 🔄 **Automatic** - No manual reloading needed
- 💾 **Versioned** - All checkpoints saved with timestamps
- 🛠️ **Configurable** - Tune training speed vs quality

---

**Ready to start training while annotating!** See `ACTIVE_LEARNING_SETUP.md` for step-by-step guide.
