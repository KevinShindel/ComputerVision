import os.path
from datetime import datetime

import cv2
from src.utils import draw_stats, extract_stats, on_change
from ultralytics import YOLO


def main():
    # weights = "../models/yolo/weights/yolo26s_tuned_best.pt"
    weights = "../models/yolo/weights/yolo26s_best.pt"

    model = YOLO(str(weights))

    # Windows-friendly webcam opening
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    controls_win = "Controls"
    view_win = "YOLO View"
    cv2.namedWindow(controls_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("conf", controls_win, 5, 10, on_change)  # dp = 1.2 default

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        conf = cv2.getTrackbarPos("conf", controls_win) / 10

        results = model.predict(
            source=frame, imgsz=640, conf=conf, verbose=False  # smaller = faster
        )

        result = results[0]
        annotated = result.plot()

        counts, total_coins, total_cents = extract_stats(result)
        annotated = draw_stats(annotated, counts, total_coins, total_cents)

        text_offset_x = 25
        text_offset_y = annotated.shape[0] - 40

        cv2.putText(
            annotated,
            'Press "s" for save result into file',
            (text_offset_x, text_offset_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.putText(
            annotated,
            'Press "q" for close window',
            (text_offset_x, text_offset_y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow(view_win, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            path = os.path.join("../output", file_name)
            cv2.imwrite(path, annotated)
            print(f"File {file_name}, saved to {path}")
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
