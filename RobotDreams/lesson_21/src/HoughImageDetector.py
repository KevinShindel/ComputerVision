from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from datetime import datetime
from src.utils import init_trackbars, detect_coins, recognize_coin, resize_keep_aspect


def main():
    course_work_dir = Path(__file__).resolve().parents[1]   # .../course_work
    test_img = course_work_dir / 'data' / 'test' / "euro_coins_example.jpg"

    parser = ArgumentParser(
        prog=__name__,
        description='YOLO Image Detector',
    )
    parser.add_argument('file_name',
                        type=str,
                        nargs='?',
                        default=test_img,
                        help='Path to image...')

    # Parse arguments
    args = parser.parse_args()
    # Set up arguments
    img_path: str = args.file_name

    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    orig_img = resize_keep_aspect(orig_img)

    controls_win = "Controls"
    view_win = "View"

    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(view_win, (800, 800))

    init_trackbars(controls_win)

    while True:
        img = orig_img.copy()

        total_money = 0
        total_coins = 0
        circles = detect_coins(img, controls_win)

        if circles is not None:
            circles = circles[0].astype(np.float32)

            total_coins = len(circles)
            max_r = np.max(circles[:, 2])

            for x, y, r in circles:
                label, value, area = recognize_coin(r, max_r)
                total_money += value

                cx, cy, cr = int(round(x)), int(round(y)), int(round(r))
                cv2.circle(img, (cx, cy), cr, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 2, (0, 255, 255), 3)

        euros = total_money // 100
        cents = total_money % 100

        cv2.putText(img, f'Total Euros: {euros}, cents: {cents}.', (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(img, f'Coins found: {total_coins}', (20, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)


        text_offset_x = 25
        text_offset_y = img.shape[0] - 40

        cv2.putText(img, 'Press "s" for save result into file', (text_offset_x, text_offset_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(img, 'Press "q" for close window', (text_offset_x, text_offset_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(view_win, img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            cv2.imwrite(file_name, img)
        else:
            continue

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
