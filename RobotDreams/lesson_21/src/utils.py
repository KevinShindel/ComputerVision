from collections import Counter
import cv2
import numpy as np
import dataclasses


@dataclasses.dataclass(frozen=True)
class CoinData:
    min_k: float
    max_k: float
    label: str
    value: int

    def matches(self, k: float) -> bool:
        return self.min_k <= k < self.max_k


EURO_CENT_1 = CoinData(0.35, 0.46, "1 cent", 1)
EURO_CENT_2 = CoinData(0.46, 0.6, "2 cent", 2)
EURO_CENT_5 = CoinData(0.6, 0.72, "5 cent", 5)
EURO_CENT_10 = CoinData(0.46, 0.60, "10 cent", 10)
EURO_CENT_20 = CoinData(0.72, 0.8, "20 cent", 20)
EURO_CENT_50 = CoinData(0.9, 0.96, "50 cent", 50)
EURO_1 = CoinData(0.8, 0.9, "1 euro", 100)
EURO_2 = CoinData(0.95, float('+inf'), "2 euro", 200)

EURO_COINS = (
    EURO_CENT_1,
    EURO_CENT_2,
    EURO_CENT_5,
    EURO_CENT_10,
    EURO_CENT_20,
    EURO_CENT_50,
    EURO_1,
    EURO_2,
)

mapping = {
    0: '1 cent',
    1: '2 cent',
    2: '5 cent',
    3: '10 cent',
    4: '20 cent',
    5: '50 cent',
    6: '1 euro',
    7: '2 euro',
}

# Coin values in euro cents
COIN_VALUES_CENTS = {
    "1 cent": 1,
    "2 cent": 2,
    "5 cent": 5,
    "10 cent": 10,
    "20 cent": 20,
    "50 cent": 50,
    "1 euro": 100,
    "2 euro": 200,
}


def normalize_coin_name(name: str) -> str:
    """
    Normalize model class names so they match the keys in COIN_VALUES_CENTS.
    """
    name = name.strip().lower()
    name = name.replace("€", " euro")
    name = " ".join(name.split())  # remove duplicate spaces

    # Optional alias handling
    aliases = {
        "1euro": "1 euro",
        "2euro": "2 euro",
        "1 cent coin": "1 cent",
        "2 cent coin": "2 cent",
        "5 cent coin": "5 cent",
        "10 cent coin": "10 cent",
        "20 cent coin": "20 cent",
        "50 cent coin": "50 cent",
    }
    return aliases.get(name, name)


def format_money(total_cents: int) -> float:
    euros = total_cents / 100.0
    return euros


def extract_stats(result):
    """
    Build statistics from one Ultralytics result object.
    """
    counts = Counter()
    total_cents = 0

    if result.boxes is not None and len(result.boxes) > 0:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for cls_id in class_ids:
            class_name = result.names[int(cls_id)]
            normalized_name = normalize_coin_name(class_name)
            counts[normalized_name] += 1
            total_cents += COIN_VALUES_CENTS.get(normalized_name, 0)

    total_coins = sum(counts.values())
    return counts, total_coins, total_cents


def draw_stats(frame, counts, total_coins, total_cents):
    """
    Draw summary statistics on top of the annotated frame.
    """
    lines = [
        f"Total coins: {total_coins}",
        f"Total money: {format_money(total_cents)}",
    ]

    if counts:
        lines.append("Detected:")
        for name, count in sorted(counts.items()):
            lines.append(f"  {name}: x{count}")
    else:
        lines.append("Detected: none")

    x, y = 10, 30
    line_height = 28

    # Background box for readability
    max_width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        max_width = max(max_width, w)

    box_w = max_width + 20
    box_h = line_height * len(lines) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 25), (x - 5 + box_w, y - 25 + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame

def crop_coin(image_bgr, x, y, r, pad_ratio=0.15):
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    pad = int(round(r * pad_ratio))
    r_pad = r + pad

    x1 = max(0, x - r_pad)
    y1 = max(0, y - r_pad)
    x2 = min(image_bgr.shape[1], x + r_pad)
    y2 = min(image_bgr.shape[0], y + r_pad)

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

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

def predict_all_coins(image_path, model, image_size=(128, 128), scale01=False):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    circles = detect_coins(image_bgr)
    if circles is None or len(circles) == 0:
        return image_bgr, [], Counter(), 0

    circles = circles[0].astype(np.float32)

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
        total_cents += COIN_VALUES_CENTS[pred_name]

    return image_bgr, predictions, counts, total_cents

def draw_predictions(image_bgr, predictions, total_cents):
    out = image_bgr.copy()

    for pred in predictions:
        x = int(round(pred["x"]))
        y = int(round(pred["y"]))
        r = int(round(pred["r"]))
        label = f'{pred["class_name"]} ({pred["confidence"]:.1%})'

        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 255, 255), 3)
        cv2.putText(
            out,
            label,
            (x - 40, y - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    euros = total_cents // 100
    cents = total_cents % 100
    cv2.putText(
        out,
        f"Total: {euros} euro {cents} cent",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    return out

def resize_keep_aspect(img, out_size=800):
    """
    Center-crop to a square (based on min(h, w)) and resize to out_size x out_size.
    This preserves circular shapes better than warping the whole image to a square.
    """
    if img is None:
        raise ValueError("img is None. Check your cv2.imread path.")

    h, w = img.shape[:2]
    side = min(h, w)

    # center crop coordinates
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    cropped = img[y1:y1 + side, x1:x1 + side]

    # Resize to requested output size
    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
    out = cv2.resize(cropped, (out_size, out_size), interpolation=interp)
    return out

def recognize_coin(r: float, max_r: float):
    max_area = 3.14159 * (max_r ** 2)
    area = 3.14159 * (r ** 2)
    coin_k = area / max_area

    for coin in EURO_COINS:
        if coin.matches(coin_k):
            return coin.label, coin.value, area

    return "unknown", 0, area




def on_change(_):
    pass

def smooth_circles(prev_circles, curr_circles, dist_threshold=35.0, alpha=0.85):
    if prev_circles is None or len(prev_circles) == 0:
        return curr_circles

    smoothed = []
    used_prev = set()

    for cx, cy, cr in curr_circles:
        best_i = -1
        best_d = 1e9

        for i, (px, py, pr) in enumerate(prev_circles):
            if i in used_prev:
                continue
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i != -1 and best_d < dist_threshold:
            px, py, pr = prev_circles[best_i]
            used_prev.add(best_i)
            smoothed.append((
                alpha * px + (1 - alpha) * cx,
                alpha * py + (1 - alpha) * cy,
                alpha * pr + (1 - alpha) * cr,
            ))
        else:
            smoothed.append((cx, cy, cr))

    return np.array(smoothed, dtype=np.float32)



def init_trackbars(controls_win):
    # Trackbars live in the controls window
    cv2.createTrackbar("dp_x10", controls_win, 12, 40, on_change)  # dp = 1.2 default
    cv2.createTrackbar("minDist", controls_win, 90, 100, on_change)
    cv2.createTrackbar("param1", controls_win, 80, 300, on_change)
    cv2.createTrackbar("param2", controls_win, 45, 200, on_change)
    cv2.createTrackbar("minRadius", controls_win, 20, 400, on_change)
    cv2.createTrackbar("maxRadius", controls_win, 90, 400, on_change)  # max bigger than value!
    cv2.createTrackbar("clipLimit", controls_win, 3, 5, on_change)  # max bigger than value!


def detect_coins(img, controls_win = 'Control'):
    dp = cv2.getTrackbarPos("dp_x10", controls_win) / 10.0
    dp = max(dp, 1.0)

    minDist = cv2.getTrackbarPos("minDist", controls_win)
    param1 = cv2.getTrackbarPos("param1", controls_win)
    param2 = cv2.getTrackbarPos("param2", controls_win)
    minRadius = cv2.getTrackbarPos("minRadius", controls_win)
    maxRadius = cv2.getTrackbarPos("maxRadius", controls_win)
    clipLimit = cv2.getTrackbarPos("clipLimit", controls_win)

    # Ensure consistent ordering
    if maxRadius > 0 and minRadius > maxRadius:
        minRadius, maxRadius = maxRadius, minRadius

    gamma = 0.8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    gray_gamma = cv2.LUT(gray, lut)

    clahe = cv2.createCLAHE(clipLimit=max(1, clipLimit),
                            tileGridSize=(8, 8)).apply(gray_gamma)  # ClipLimit = 2
    gray_blur = cv2.GaussianBlur(clahe, (9, 9), 2.5)

    # HoughCircles Detection
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,  # 1.2
        minDist=max(1, minDist),  # 60
        param1=max(1, param1),  # 175
        param2=max(1, param2),  # 35
        minRadius=minRadius,  # 20
        maxRadius=maxRadius,  # 120
    )
    return circles