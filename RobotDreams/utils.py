import os
from typing import Sequence
from urllib.request import urlretrieve
import cv2
import matplotlib.pyplot as plt

models = (
    # DASiamRPN model
    ('https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=1', 'models/dasiamrpn'),
    ('https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=1', 'models/dasiamrpn'),
    ('https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=1', 'models/dasiamrpn'),
    # NanoTrack model
    ('https://raw.githubusercontent.com/HonglinChu/SiamTrackers/refs/heads/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx', 'models/nanotrack'),
    ('https://raw.githubusercontent.com/HonglinChu/SiamTrackers/refs/heads/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx', 'models/nanotrack'),
    # ViTTrack model
    ('https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx', 'models/vittrack'),
    # YuNet model
    ('https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 'models/yunet'),
    # OpenCVDnn model
    ('https://raw.githubusercontent.com/spmallick/learnopencv/refs/heads/master/FaceDetectionComparison/models/deploy.prototxt', 'models/opencvdnn'),
    ('https://raw.githubusercontent.com/spmallick/learnopencv/refs/heads/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel', 'models/opencvdnn'),
    # FaceRecognizerSF
    ('https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx', 'models/facerecognizer'),
)


images = [
    # TODO: find more images from https://github.com/opencv/opencv/tree/4.x/samples/data repo!
    ('https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/chessboard.png', 'images'),
]

videos = [
    # TODO: find more videos from https://github.com/opencv/opencv/tree/4.x/samples/data repo!
    ('https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4', 'videos'),
    ('https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/vtest.avi', 'videos'),
]

def fetch_data(urls):
    """ Fetches data from the given URLs and saves them to the specified directory."""
    for idx, (url, path) in enumerate(urls): # TODO: make this async for better performance
        print(f'Processed {idx / len(urls) * 100:.2f}%: {url}')
        os.makedirs(path, exist_ok=True)
        filename = os.path.basename(url.split('?')[0]) # Get the filename from the URL
        save_path = os.path.join(path, filename)
        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            urlretrieve(url, save_path)
        else:
            print(f"{filename} already exists. Skipping download.")

def draw_rectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def display_rectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    draw_rectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


def draw_text(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def select_roi(frame) -> Sequence[float]:
    overlay = frame.copy()
    win_name = "ROI Instructions"

    msg1 = "Please select ROI by mouse."
    msg2 = "Press ENTER/SPACE to confirm or ESC to cancel."

    # Draw a filled rectangle as a text background (top-left)
    cv2.rectangle(overlay, (10, 10), (980, 90), (0, 0, 0), thickness=-1)
    cv2.putText(overlay, msg1, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, msg2, (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Drag a rectangle with the mouse, press ENTER/SPACE to confirm, or ESC to cancel
    object_template = cv2.selectROI(win_name,
                                    overlay,
                                    showCrosshair=True,
                                    fromCenter=False)
    cv2.destroyWindow("Select ROI")

    return object_template

if __name__ == "__main__":
    for batch in (models, images, videos):
        print(f'fetching data... {str(batch[0][1])}...')
        fetch_data(batch)