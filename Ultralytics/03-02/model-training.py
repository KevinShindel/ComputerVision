from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../models/yolo11n.pt")

    model.train(
        data="data.yaml",
        batch=16,
        workers=1,
        epochs=5,
    )

    model.predict(
        source="../media/apples_video.mov",
        show=True,
        line_width=2,
    )

# TODO: create a notebook for this module
