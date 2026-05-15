import tkinter as tk
from argparse import ArgumentParser
from pathlib import Path

import cv2
from ultralytics import YOLO


def monitor_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height


def main():
    course_work_dir = Path(__file__).resolve().parents[1]  # .../course_work
    weights = course_work_dir / "models" / "yolo" / "weights" / "yolo26s_best.pt"
    test_img = course_work_dir / "data" / "test" / "euro_coins_example_2.jpg"

    # Define Parser
    parser = ArgumentParser(
        prog=__name__,
        description="YOLO Image Detector",
    )
    parser.add_argument(
        "file_name", type=str, nargs="?", default=test_img, help="Path to image..."
    )

    parser.add_argument(
        "-s",
        "--save",
        default=False,
        type=bool,
        required=False,
        # action='save_image',
        help="Save final image ? ",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="Set up verbosity level (0, 1)",
    )

    # Parse arguments
    args = parser.parse_args()
    # Set up arguments
    img_path: str = args.file_name
    save_img: bool = args.save
    verbose_lvl: int = args.verbose

    eval_model = YOLO(str(weights))  # Load model

    results = eval_model.predict(  # Evaluation
        source=str(img_path),
        show=False,
        save=save_img,
        save_txt=True,
        save_conf=True,
        project=str(course_work_dir),
        name="output",
        exist_ok=True,
    )

    result = results[0]
    annotated = result.plot()

    window_name = "YOLO"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    screen_w, screen_h = monitor_resolution()

    h, w, channel_nbr = annotated.shape
    # img get w of screen and adapt h
    h = h * (screen_w / w)
    w = screen_w
    if h > screen_h:  # if img h still too big
        # img get h of screen and adapt w
        w = w * (screen_h / h)
        h = screen_h
    w, h = w * 0.9, h * 0.9  # because you don't want it to be that big, right ?
    w, h = int(w), int(h)  # you need int for the cv2.resize

    annotated = cv2.resize(annotated, (w, h))

    cv2.resizeWindow(window_name, w, h)

    text_offset_x = 25
    text_offset_y = annotated.shape[0] - 40

    cv2.putText(
        annotated,
        'Press "q" for close window',
        (text_offset_x, text_offset_y + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    cv2.imshow(window_name, annotated)
    cv2.waitKey(0)  # wait forever until a key is pressed
    cv2.destroyAllWindows()

    if verbose_lvl:
        print("Saved to:", result.save_dir)


if __name__ == "__main__":
    main()
