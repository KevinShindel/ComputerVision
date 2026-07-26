from ultralytics import solutions

model = 'new-model.pt'

inf = solutions.Inference(
    model=model,  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)

inf.inference()

# Make sure to run the file using command `streamlit run path/to/file.py`