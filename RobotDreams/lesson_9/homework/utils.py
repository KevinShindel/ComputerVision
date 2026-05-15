import cv2
import numpy as np


class OpenCVDnnFaceDetector:
    """
    OpenCV DNN face detector wrapper (SSD-style).
    Output format (per face): (x, y, w, h, confidence)
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        backend: int = cv2.dnn.DNN_BACKEND_DEFAULT,
        target: int = cv2.dnn.DNN_TARGET_CPU,
        input_size: tuple[int, int] = (640, 640),
        mean: tuple[float, float, float] = (104.0, 177.0, 123.0),
        swap_rb: bool = False,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.input_size = input_size
        self.mean = mean
        self.swap_rb = swap_rb

        if config_path:
            if config_path.endswith(".prototxt"):
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            else:
                self.net = cv2.dnn.readNet(model_path, config_path)
        else:
            self.net = cv2.dnn.readNet(model_path)

        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

        self.out: list[tuple[int, int, int, int, float]] = []

    def __call__(
        self,
        bgr: np.ndarray,
        conf_threshold: float = 0.2,
        nms_threshold: float = 0.4,
        use_nms: bool = True,
    ):
        h, w = bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            bgr,
            scalefactor=1.0,
            size=self.input_size,
            mean=self.mean,
            swapRB=self.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        det = self.net.forward()  # (1, 1, N, 7)

        boxes: list[list[int]] = []
        scores: list[float] = []

        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < conf_threshold:
                continue

            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue

            boxes.append([x1, y1, bw, bh])
            scores.append(conf)

        out: list[tuple[int, int, int, int, float]] = []
        if use_nms and boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            if len(idxs) > 0:
                for k in idxs.flatten().tolist():
                    x, y, bw, bh = boxes[k]
                    out.append((x, y, bw, bh, float(scores[k])))
        else:
            for (x, y, bw, bh), conf in zip(boxes, scores):
                out.append((x, y, bw, bh, float(conf)))

        self.out = out
        print("Number of detected faces:", len(self.out))
        return self

    def draw_faces(self, rgb_img: np.ndarray):
        result = np.copy(rgb_img)
        for x, y, w, h, confidence in self.out:
            color = tuple(int(v) for v in np.random.randint(0, 255, size=3))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result,
                str(np.round(confidence, 2)),
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        return result

    def get_result(self):
        return self.out

    def get_total_observed(self):
        return len(self.out)


class YunetFaceDetector:
    """
    YuNet (cv2.FaceDetectorYN) wrapper.
    Output format (per face): (x, y, w, h, confidence)
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
    ):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # Create with a dummy size; real input size is set per image in `__call__`.
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

        self.out: list[tuple[int, int, int, int, float]] = []

    def __call__(self, bgr_img: np.ndarray):
        h, w = bgr_img.shape[:2]

        # Critical: must match current image size (otherwise results can be empty/incorrect).
        self.detector.setInputSize((w, h))

        retval, faces = self.detector.detect(bgr_img)
        if faces is None or len(faces) == 0:
            self.out = []
            print("Number of detected faces:", 0)
            return self

        out: list[tuple[int, int, int, int, float]] = []
        for row in faces:
            x, y, fw, fh = row[0:4]
            conf = float(row[14])

            x_i = int(round(x))
            y_i = int(round(y))
            w_i = int(round(fw))
            h_i = int(round(fh))

            if w_i <= 0 or h_i <= 0:
                continue

            # Clamp to image bounds
            x_i = max(0, min(x_i, w - 1))
            y_i = max(0, min(y_i, h - 1))
            w_i = min(w_i, w - x_i)
            h_i = min(h_i, h - y_i)

            out.append((x_i, y_i, w_i, h_i, conf))

        self.out = out
        print("Number of detected faces:", len(self.out))
        return self

    def draw_faces(self, rgb_img: np.ndarray):
        result = np.copy(rgb_img)
        for x, y, w, h, confidence in self.out:
            color = tuple(int(v) for v in np.random.randint(0, 255, size=3))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result,
                str(np.round(confidence, 2)),
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        return result

    def get_result(self):
        return self.out

    def get_total_observed(self):
        return len(self.out)


class HaarDetector:

    def __init__(self, scaleFactor=1.2, minNeighbors=20):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(casc_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from: {casc_path}")
        self.out = []

    def __call__(self, rgb_img: np.ndarray, *, minSize=(30, 30), maxSize=None):
        # 1) Store the current image so draw_faces uses the same one
        self.img = rgb_img

        # 2) Haar works on grayscale
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # 3) Normalize to a python list of tuples
        if faces is None or len(faces) == 0:
            self.out = []
        else:
            self.out = [tuple(map(int, box)) for box in faces]

        print("Number of detected faces:", len(self.out))
        return self

    def draw_faces(self):
        if self.img is None:
            raise RuntimeError("No image set. Call the detector first.")
        result = np.copy(self.img)
        for x, y, w, h in self.out:
            color = (
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
                int(np.random.randint(0, 255)),
            )
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        return result
