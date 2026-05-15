import cv2
from kalman_filter import CVKalmanMotionModel


def clamp_point(pt, w, h):
    x, y = pt
    return (int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y))))


def draw_trajectory(frame, points, color=(0, 255, 255), thickness=2):
    if len(points) < 2:
        return
    for a, b in zip(points[:-1], points[1:]):
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
    # mark end
    cv2.circle(frame, points[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)


def predict_future_xy(kf: CVKalmanMotionModel, seconds_ahead: float, fps: float):
    """
    Predict future positions assuming constant velocity for `seconds_ahead`.
    Note: OpenCV KalmanFilter has no cheap "peek" predict without advancing state,
    so we clone the filter state and run predict() on the clone.
    """
    steps = int(max(1, round(seconds_ahead * fps)))

    kf2 = CVKalmanMotionModel(dt=kf.dt)
    kf2.kf.transitionMatrix = kf.kf.transitionMatrix.copy()
    kf2.kf.measurementMatrix = kf.kf.measurementMatrix.copy()
    kf2.kf.processNoiseCov = kf.kf.processNoiseCov.copy()
    kf2.kf.measurementNoiseCov = kf.kf.measurementNoiseCov.copy()
    kf2.kf.errorCovPost = kf.kf.errorCovPost.copy()

    # copy state (posterior) into clone
    kf2.kf.statePost = kf.kf.statePost.copy()
    kf2.kf.statePre = kf.kf.statePre.copy()
    kf2.initialized = True

    traj = []
    for _ in range(steps):
        x, y = kf2.predict()
        traj.append((x, y))
    return traj


def track_and_draw(video_path: str, seconds_ahead: float = 3.0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 15)  # Skip first 15 frames to avoid the intro
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open `video_path`: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / float(fps)

    kf = CVKalmanMotionModel(dt=dt, process_noise=1e-2, meas_noise=1e-1)

    # Placeholder: you already have ROI + TrackerNano.
    ok, frame = cap.read()
    object_template = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    # You just need to produce per-frame bbox = (x, y, w, h) or None.
    backbone = '../../models/nanotrack_backbone_sim.onnx'
    neckhead = '../../models/nanotrack_head_sim.onnx'

    params = cv2.TrackerNano_Params()
    params.backbone = str(backbone)
    params.neckhead = str(neckhead)
    params.backend = cv2.dnn.DNN_BACKEND_OPENCV
    params.target = cv2.dnn.DNN_TARGET_CPU

    tracker = cv2.TrackerNano.create(params)
    tracker.init(frame, object_template)
    cap.release()

    initialized = True

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 15)  # Skip first 15 frames to avoid the intro

    while True:
        ok, frame = cap.read()

        if not ok:
            break
        h, w = frame.shape[:2]

        ok, bbox = tracker.update(frame)
        if not ok:
            bbox = None

        if bbox is not None:
            x, y, bw, bh = bbox
            cx = float(x + bw * 0.5)
            cy = float(y + bh * 0.5)

            if not initialized:
                kf.init(cx, cy, vx=0.0, vy=0.0)
                initialized = True
            else:
                kf.predict()
                kf.update(cx, cy)

            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        else:
            if initialized:
                kf.predict()

        if initialized:
            future = predict_future_xy(kf, seconds_ahead=seconds_ahead, fps=fps)
            pts = [clamp_point(p, w, h) for p in future]
            draw_trajectory(frame, pts, color=(0, 255, 255), thickness=2)

            cv2.putText(
                frame,
                f"KF future: {seconds_ahead:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Tracking + Kalman trajectory", frame)
        # add small delay and exit on ESC key
        # cv2.waitKey(500)
        if cv2.waitKey(250) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_and_draw(video_path=r"../../video/6km_nissan_z340.mp4", seconds_ahead=1.0)
