# Active Learning - Quick Start Checklist

## ✓ What Was Created

- `ml_backend/active_learning/` - Complete Active Learning module
  - `trainer.py` - YOLO fine-tuning logic
  - `data_converter.py` - Label Studio → YOLO format converter
  - `__init__.py` - Module exports
  - `README.md` - Technical documentation

- `ml_backend/model.py` - UPDATED with Active Learning support
- `ml_backend/.env.example` - UPDATED with AL config options
- `ACTIVE_LEARNING_SETUP.md` - Comprehensive setup & troubleshooting guide
- `ACTIVE_LEARNING_SUMMARY.md` - Implementation overview

## ✓ Files NOT Modified (As Requested)

- ✅ All notebooks remain untouched (`LabelStudio.ipynb`, `YoloTraining.ipynb`, etc.)
- ✅ Everything is in separate `active_learning/` folder
- ✅ Can be disabled with one config flag

---

## 🚀 To Enable Active Learning

### Step 1️⃣: Update `.env`

Copy these lines to your `.env` file:

```ini
# Active Learning settings
ACTIVE_LEARNING_ENABLED=true
AL_EPOCHS=3
AL_BATCH_SIZE=4
AL_IMGSZ=640
AL_DEVICE=cpu
AL_TRAINING_DIR=./al-training
```

### Step 2️⃣: Enable in LabelStudio UI

1. Open LabelStudio
2. Go to your Project
3. **Settings** → **Model** → **Configuration**
4. ✅ Check: **"Start model training on annotation submission"**
5. Click **Save**

### Step 3️⃣: Start Backend

```bash
cd ml_backend
python server.py
```

### Step 4️⃣: Connect Model in LabelStudio

1. LabelStudio Project → **Model** → **Connect**
2. Enter: `http://localhost:9090`
3. Select model → **Connect**

### Step 5️⃣: Start Annotating!

Just annotate as normal. Training happens automatically after each submission.

---

## 📊 Monitor Training

### View Training Data
```bash
ls -la al-training/
# You'll see:
# - images/      (training images)
# - labels/      (YOLO annotations)
# - checkpoints/ (trained models)
# - data.yaml    (YOLO config)
```

### Check Backend Logs
Look for:
```
[INFO] Received N annotations
[INFO] Added N training samples
[INFO] ✓ Training completed | checkpoint=...
```

### View Checkpoint Info
```bash
cat al-training/checkpoints/yolo_al_ep3_*.json
```

---

## ⚙️ Tuning (Optional)

### For Faster Training
```ini
AL_EPOCHS=2
AL_BATCH_SIZE=2
AL_IMGSZ=416
```

### For GPU Training
```ini
AL_DEVICE=cuda
AL_BATCH_SIZE=8
AL_IMGSZ=640
```

### For Minimal Resource Usage
```ini
AL_EPOCHS=1
AL_BATCH_SIZE=1
AL_IMGSZ=416
AL_DEVICE=cpu
```

---

## 🔧 Troubleshooting

**Model not updating?**
- Check logs for "Received N annotations"
- Verify annotations have bboxes
- Ensure `ACTIVE_LEARNING_ENABLED=true`

**Out of memory?**
- Reduce `AL_BATCH_SIZE` → 2 or 1
- Reduce `AL_IMGSZ` → 416

**Training too slow?**
- Reduce `AL_EPOCHS` → 1 or 2
- Reduce `AL_IMGSZ` → 416

See **ACTIVE_LEARNING_SETUP.md** for detailed troubleshooting.

---

## 📚 Documentation

1. **ACTIVE_LEARNING_SETUP.md** ← Start here (complete guide)
2. **ACTIVE_LEARNING_SUMMARY.md** ← Overview & architecture
3. **ml_backend/active_learning/README.md** ← Technical details

---

## 🎯 How It Works (30 seconds)

```
User annotates
    ↓
Submit annotations
    ↓
/train endpoint called
    ↓
YOLO fine-tunes on new data (few epochs)
    ↓
Save checkpoint
    ↓
Next predictions use improved model
    ↓
Repeat → Model gets better! 🚀
```

---

## ✨ Key Benefits

- **Incremental learning** - Model improves as you annotate
- **Automatic** - No manual steps needed
- **Non-blocking** - Training happens in background
- **Configurable** - Tune for speed or quality
- **Versioned** - All checkpoints saved
- **Easy disable** - Set `ACTIVE_LEARNING_ENABLED=false` if needed

---

## ❓ Questions?

See documentation files above or check logs for errors.

---

**Ready to go!** 🚀 Follow the 5 steps above to enable Active Learning.
