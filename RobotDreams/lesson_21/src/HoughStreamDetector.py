from datetime import datetime

import cv2
import numpy as np
from src.utils import detect_coins, init_trackbars, recognize_coin


def main():
    # Windows-friendly webcam opening
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FPS, 10)

    controls_win = "Controls"
    view_win = "View"

    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)
    init_trackbars(controls_win)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        total_money = 0
        total_coins = 0
        circles = detect_coins(frame, controls_win)

        if circles is not None:
            circles = circles[0].astype(np.float32)

            total_coins = len(circles)
            max_r = np.max(circles[:, 2])

            for x, y, r in circles:
                label, value, area = recognize_coin(r, max_r)
                total_money += value

                cx, cy, cr = int(round(x)), int(round(y)), int(round(r))
                cv2.circle(frame, (cx, cy), cr, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), 3)

        euros = total_money // 100
        cents = total_money % 100

        cv2.putText(
            frame,
            f"Total Euros: {euros}, cents: {cents}.",
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"Coins found: {total_coins}",
            (20, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        text_offset_x = 25
        text_offset_y = frame.shape[0] - 40

        cv2.putText(
            frame,
            'Press "s" for save result into file',
            (text_offset_x, text_offset_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            'Press "q" for close window',
            (text_offset_x, text_offset_y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow(view_win, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            cv2.imwrite(file_name, frame)
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
