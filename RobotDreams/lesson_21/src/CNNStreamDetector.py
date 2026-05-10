from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

mapping = {
    0: '1 Cent',
    1: '2 Cent',
    2: '5 Cent',
    3: '10 Cent',
    4: '20 Cent',
    5: '50 Cent',
    6: '1 Euro',
    7: '2 Euro',
}

coin_to_cents = {
    '1 Cent': 1,
    '2 Cent': 2,
    '5 Cent': 5,
    '10 Cent': 10,
    '20 Cent': 20,
    '50 Cent': 50,
    '1 Euro': 100,
    '2 Euro': 200,
}

def detect_coins_hough(image_bgr):
    # TODO: Try use R-CNN instead HoughCircles
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gamma = 0.8
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    gray_gamma = cv2.LUT(gray, lut)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray_gamma)
    gray_blur = cv2.medianBlur(clahe, 5)

    circles = cv2.HoughCircles(gray_blur,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=60,
                               param1=175,
                               param2=35,
                               minRadius=20,
                               maxRadius=120,
                               )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    return circles.tolist()   # [[x, y, r], ...]

def crop_coin(image_bgr, x, y, r, pad_ratio=0.15):
    pad = int(r * pad_ratio)
    r_pad = r + pad

    x1 = max(0, x - r_pad)
    y1 = max(0, y - r_pad)
    x2 = min(image_bgr.shape[1], x + r_pad)
    y2 = min(image_bgr.shape[0], y + r_pad)

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    # Optional: mask background outside the circle
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cx = x - x1
    cy = y - y1
    cv2.circle(mask, (cx, cy), r, 255, -1)

    white_bg = np.full_like(crop, 255)
    crop = np.where(mask[..., None] == 255, crop, white_bg)

    return crop

def preprocess_crop_for_model(crop_bgr, image_size=(128, 128), scale01=False):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, image_size, interpolation=cv2.INTER_AREA)
    x = crop_rgb.astype("float32")

    # Use this only if your model was trained on 0..1 inputs
    if scale01:
        x = x / 255.0

    return x

def predict_all_coins(image_bgr, model, image_size=(128, 128), scale01=False):

    circles = detect_coins_hough(image_bgr)
    if not circles:
        return image_bgr, [], Counter(), 0

    crops = []
    kept_circles = []

    for x, y, r in circles:
        crop = crop_coin(image_bgr, x, y, r)
        if crop is None:
            continue

        x_model = preprocess_crop_for_model(crop, image_size=image_size, scale01=scale01)
        crops.append(x_model)
        kept_circles.append((x, y, r))

    if not crops:
        return image_bgr, [], Counter(), 0

    batch = np.stack(crops, axis=0)
    probs = model.predict(batch, verbose=0)

    predictions = []
    counts = Counter()
    total_cents = 0

    for (x, y, r), p in zip(kept_circles, probs):
        pred_idx = int(np.argmax(p))
        pred_name = mapping[pred_idx]
        conf = float(p[pred_idx])

        predictions.append({
            "x": x,
            "y": y,
            "r": r,
            "class_id": pred_idx,
            "class_name": pred_name,
            "confidence": conf,
        })

        counts[pred_name] += 1
        total_cents += coin_to_cents[pred_name]

    return image_bgr, predictions, counts, total_cents

def draw_predictions(image_bgr, predictions, total_cents):
    out = image_bgr.copy()

    for pred in predictions:
        x, y, r = pred["x"], pred["y"], pred["r"]
        label = f'{pred["class_name"]} ({pred["confidence"]:.1%})'

        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 255, 255), 3)
        cv2.putText(out, label, (x - 40, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    euros = total_cents // 100
    cents = total_cents % 100
    cv2.putText(out, f"Total: {euros} euro {cents} cent",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out

def main():
    # Example usage
    course_work_dir = Path(r"C:\Users\username\Projects\Computer-Vision-v2\course_work")
    weights = course_work_dir / "models" / "tensorflow" / "weights" / "best_model.keras"
    model = tf.keras.models.load_model(weights, compile=False)

    # Windows-friendly webcam opening
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    view_win = "CNN Coin Detector"
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break


        image_bgr, predictions, counts, total_cents = predict_all_coins(
            frame,
            model,
            image_size=(128, 128),
            scale01=False,   # change to True if training used x/255 outside the model
        )

        annotated = draw_predictions(image_bgr, predictions, total_cents)

        cv2.imshow(view_win, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
