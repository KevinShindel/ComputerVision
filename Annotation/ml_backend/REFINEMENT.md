# Bbox Refinement Implementation

## Problem
YOLO v8 detection bounding boxes are sometimes shifted or inaccurate. This is because:
- YOLO uses fast inference at reduced resolution (640x640 letterboxed)
- The model may not be perfectly calibrated for this specific coin dataset
- Coordinates need to be carefully scaled back to original image space

## Solution
Use SAM2 segmentation masks to refine YOLO bounding boxes:

1. **YOLO Detection**: Get initial bounding boxes
2. **SAM2 Segmentation**: Use YOLO boxes as prompts to get precise pixel-level masks
3. **Bbox Refinement**: Compute accurate bbox from the segmentation mask
4. **Two Strategies Available**:
   - `mask`: Compute bbox directly from mask contours (more aggressive correction)
   - `centroid`: Use mask center-of-mass to adjust YOLO bbox center (subtle correction)

## Files

### `bbox_refinement.py`
Core refinement logic:
- `compute_bbox_from_mask(mask)`: Extract bbox from mask contours
- `compute_mask_centroid(mask)`: Calculate weighted center of mass
- `refine_bbox_with_mask(yolo_bbox, mask, use_centroid)`: Refine single bbox
- `refine_bboxes_batch(yolo_bboxes, masks)`: Batch refinement

### `model.py`
Drop-in replacement for `model.py`:
- Uses `YoloSamBackendRefined` class
- Integrates bbox refinement into prediction pipeline
- Configurable via env var `REFINEMENT_STRATEGY`

### `server.py`
Drop-in replacement for `server.py`:
- Uses the refined model instead of standard model
- Same configuration options as original server

## Usage

### Option 1: Direct Replacement
Replace `server.py` and `model.py` with refined versions:
```bash
python server.py --host 0.0.0.0 --port 9090
```

### Option 2: Side-by-Side Testing
Run the refined server on a different port:
```bash
python server.py --host 0.0.0.0 --port 9091
```

## Configuration

Set via environment variables in `.env`:

```bash
# Refinement strategy: "mask" (default) or "centroid"
REFINEMENT_STRATEGY=mask

# Other standard options
YOLO_MODEL=models/yolo26n.pt
SAM_MODEL=models/sam2.1_b.pt
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

## Strategy Comparison

### `mask` Strategy (Default)
- **Approach**: Compute bbox directly from SAM2 mask contours
- **Pros**: Very accurate, uses full segmentation information
- **Cons**: May be too aggressive if SAM mask is slightly off
- **Use when**: YOLO bboxes are noticeably inaccurate

### `centroid` Strategy
- **Approach**: Use mask center-of-mass to adjust YOLO bbox center
- **Pros**: Subtle correction, preserves YOLO bbox size
- **Cons**: Less aggressive refinement
- **Use when**: YOLO bboxes are mostly correct but slightly misaligned

## Results

The refinement should produce:
- More accurate bounding box centers
- Better alignment with actual coin positions
- Coordinates that stay within image bounds
- Consistent labeling across multiple coins

## Troubleshooting

If results are worse after refinement:
1. Try the other refinement strategy: `REFINEMENT_STRATEGY=centroid`
2. Check SAM2 mask quality in the logs
3. Consider that YOLO bboxes might be correct despite appearing off
4. Revert to `server.py` with original `model.py` for baseline comparison
