# Active Model Architecture

## Overview

Active Learning functionality is now completely separate from the original `model.py` using a clean inheritance-based architecture.

## File Structure

```
ml_backend/
├── model.py                  # Original YoloSamBackend (UNCHANGED)
├── active_model.py           # NEW: ActiveYoloSamBackend with Active Learning
├── server.py                 # Updated to use model class based on config
├── active_learning/          # Training module
│   ├── trainer.py           # YOLO fine-tuning
│   ├── data_converter.py    # Annotation converter
│   └── __init__.py
└── ...
```

## Class Architecture

### YoloSamBackend (model.py - Original)

```python
class YoloSamBackend(LabelStudioMLBase):
    def predict(tasks) → predictions
    def fit(annotations) → {"status": "ok", ...}
```

**Features:**
- YOLO26 object detection
- SAM2.1 segmentation
- Bbox refinement
- No training on annotations

### ActiveYoloSamBackend (active_model.py - New)

```python
class ActiveYoloSamBackend(YoloSamBackend):
    def __init__():
        self.al_trainer = ActiveLearningTrainer(...)
        
    def predict(tasks) → predictions (+ auto-loads latest checkpoint)
    def fit(annotations) → trains asynchronously
```

**Extends YoloSamBackend with:**
- Incremental YOLO fine-tuning
- Auto-detection of new checkpoints
- Background training thread
- Checkpoint versioning

## How It Works

### Startup

`server.py` checks `.env` on initialization:

```python
if ACTIVE_LEARNING_ENABLED == "true":
    model_class = ActiveYoloSamBackend  # from active_model.py
else:
    model_class = YoloSamBackend  # from model.py

app = init_app(model_class=model_class, ...)
```

### Runtime Flow

**With Active Learning Enabled:**

```
User annotates
    ↓
Submit annotation
    ↓
server.py → ActiveYoloSamBackend.fit()
    ↓
    ├─ Extract annotations
    ├─ Start async training in background
    └─ Return immediately
    ↓
Background training:
    ├─ Convert to YOLO format
    ├─ Fine-tune model
    ├─ Save checkpoint
    └─ Ready for next predict()
    ↓
Next predict() call:
    ├─ Check for new checkpoint
    ├─ Auto-load if newer than current
    └─ Use improved model
```

**With Active Learning Disabled:**

```
User annotates
    ↓
Submit annotation
    ↓
server.py → YoloSamBackend.fit()
    ↓
    └─ Log and return "ok" (no training)
    ↓
predict() call:
    └─ Uses original model (unchanged)
```

## Configuration

### .env

```ini
# Enable/disable Active Learning
ACTIVE_LEARNING_ENABLED=true

# Active Learning parameters (only used if enabled)
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=640
AL_DEVICE=cpu
AL_TRAINING_DIR=./al-training
```

### Runtime Switching

You can enable/disable without code changes:

```bash
# Use Active Learning
ACTIVE_LEARNING_ENABLED=true python server.py

# Disable Active Learning
ACTIVE_LEARNING_ENABLED=false python server.py
```

## Startup Logs

**With Active Learning:**
```
[INFO] Using ActiveYoloSamBackend (Active Learning enabled)
[INFO] Active Learning enabled | epochs=3 | batch=4 | device=cpu
```

**Without Active Learning:**
```
[INFO] Using YoloSamBackend (standard)
```

## Key Design Points

✅ **No modifications to model.py**
- Original remains completely unchanged
- Can be used as-is without Active Learning

✅ **Inheritance-based extension**
- ActiveYoloSamBackend extends YoloSamBackend
- Reuses predict pipeline from parent
- Adds training via fit() override

✅ **Smart routing**
- server.py decides which class to use
- Based on .env configuration
- Happens at startup time

✅ **Independent modules**
- active_model.py has no dependencies on model.py beyond importing YoloSamBackend
- active_learning/ is completely self-contained
- Can be removed without breaking original model

## Using in Code

### Option 1: Import Specific Class

```python
# Only original model
from model import YoloSamBackend
backend = YoloSamBackend()

# Only with active learning
from active_model import ActiveYoloSamBackend
backend = ActiveYoloSamBackend()
```

### Option 2: Server (Recommended)

```bash
cd ml_backend
python server.py
# Automatically selects based on .env
```

## Methods

### YoloSamBackend.predict(tasks)
- Input: List of tasks with images
- Output: List of predictions with bboxes
- Behavior: Standard YOLO+SAM2 pipeline

### YoloSamBackend.fit(annotations)
- Input: List of annotations
- Output: {"status": "ok", "annotations_received": N}
- Behavior: Returns immediately (no training)

### ActiveYoloSamBackend.predict(tasks)
- Inherits from YoloSamBackend
- **NEW**: Checks for new checkpoint before predicting
- **NEW**: Auto-loads latest checkpoint if available
- Then: Runs parent predict()

### ActiveYoloSamBackend.fit(annotations)
- Input: List of annotations
- Output: {"status": "training_queued", "annotations_received": N, ...}
- **NEW**: Starts async training thread
- **NEW**: Trains on accumulated data
- **NEW**: Saves checkpoint with timestamp

## Switching Models

### Temporary (One-time)

```bash
ACTIVE_LEARNING_ENABLED=false python server.py
```

### Permanent (Edit .env)

```ini
ACTIVE_LEARNING_ENABLED=false
```

Then restart server.

### In Code

```python
# Always use original
from model import YoloSamBackend
model_class = YoloSamBackend

# Always use with AL
from active_model import ActiveYoloSamBackend
model_class = ActiveYoloSamBackend

app = init_app(model_class=model_class, ...)
```

## Debugging

### Check Which Model Is Loaded

Look at startup logs:
```
[INFO] Using ActiveYoloSamBackend (Active Learning enabled)
```

### Check Active Learning Settings

```bash
grep ACTIVE_LEARNING .env
# ACTIVE_LEARNING_ENABLED=true
# AL_EPOCHS=3
# AL_BATCH_SIZE=4
# ...
```

### Verify Training Is Happening

```bash
# Check training data
ls -la al-training/images/
ls -la al-training/labels/

# Check checkpoints
ls -la al-training/checkpoints/
```

### View Logs

```
[INFO] Using ActiveYoloSamBackend (Active Learning enabled)
[INFO] Active Learning enabled | epochs=3 | batch=4 | device=cpu
[INFO] YoloSamBackend ready | strategy=mask | conf=0.25 | iou=0.45
```

On annotation:
```
[INFO] Received N annotations
[INFO] Started async training thread
```

During training:
```
[INFO] Starting Active Learning training | annotations=N
[INFO] Added N training samples
[INFO] ✓ Training completed | checkpoint=...
```

## Compatibility

✅ Drop-in replacement for original model
- Same interface (predict, fit)
- Same configuration (.env variables)
- Compatible with Label Studio ML framework

✅ Can mix both in same project
- Use YoloSamBackend for standard predictions
- Use ActiveYoloSamBackend for training scenarios

## Performance

**ActiveYoloSamBackend.predict():**
- Same as YoloSamBackend (usually ~100-200ms)
- +10-50ms for checkpoint comparison (negligible)

**ActiveYoloSamBackend.fit():**
- Returns immediately (<1ms)
- Training happens async in background

## Troubleshooting

**Q: Model.py is being used but I set ACTIVE_LEARNING_ENABLED=true?**
A: Check .env is being loaded. server.py reads it at startup.

**Q: Can I use both models?**
A: Yes! Import and create instances separately:
```python
from model import YoloSamBackend
from active_model import ActiveYoloSamBackend

model1 = YoloSamBackend()
model2 = ActiveYoloSamBackend()
```

**Q: Will disabling AL break anything?**
A: No! YoloSamBackend is the original, unchanged code.

**Q: Can I inspect which class is being used?**
A: Check startup logs or inspect `model_class.__name__`.

---

**Summary:** Clean separation of concerns with inheritance-based extension. Original untouched, Active Learning fully optional. 🎯
