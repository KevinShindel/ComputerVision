from ultralytics import YOLO

model = YOLO("yolov8n.pt")

yaml_path = r'/course_work/data/coin_yolo_split/data.yaml'

model.train(
    data=yaml_path,
    epochs=1,
    imgsz=128,
    batch=1,
    device="cpu"
)