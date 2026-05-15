from argparse import ArgumentParser
from pathlib import Path
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import draw_predictions, predict_all_coins, init_trackbars


def main():
    course_work_dir = Path(__file__).resolve().parents[1]  # .../course_work
    weights = course_work_dir / 'models' / 'tensorflow' / 'weights' / 'best_model.keras'
    test_img = course_work_dir / 'data' / 'test' / "euro_coins_example.jpg"

    parser = ArgumentParser(
        prog=__name__,
        description='CNN Image Detector',
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

    # Example usage
    model = tf.keras.models.load_model(weights, compile=False)

    controls_win = 'Control'
    cv2.namedWindow(controls_win, cv2.WINDOW_NORMAL)
    init_trackbars(controls_win)

    image_bgr, predictions, counts, total_cents = predict_all_coins(
        img_path,
        model,
        image_size=(128, 128),
        scale01=False,   # change to True if training used x/255 outside the model
    )

    print("Detected coins:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    annotated = draw_predictions(image_bgr, predictions, total_cents)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
