import numpy as np
import cv2


class CVKalmanMotionModel:
    """
    2D constant-velocity Kalman filter:
    state = [x, y, vx, vy]^T
    measurement = [x, y]^T
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 1e-2, meas_noise: float = 1e-1):
        self.dt = float(dt)

        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array(
            [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32
        )

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32
        )

        self.kf.processNoiseCov = (process_noise * np.eye(4, dtype=np.float32))
        self.kf.measurementNoiseCov = (meas_noise * np.eye(2, dtype=np.float32))
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.kf.statePost = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()  # shape: (4, 1)
        return pred[0, 0].item(), pred[1, 0].item()

    def update(self, x: float, y: float) -> tuple[float, float]:
        z = np.array([[x], [y]], dtype=np.float32)  # shape: (2, 1)
        est = self.kf.correct(z)  # shape: (4, 1)
        return est[0, 0].item(), est[1, 0].item()


if __name__ == '__main__':
    kf_instance = CVKalmanMotionModel()
    kf_instance.init(0.0, 0.0, 1.0, 1.0)

    for _ in range(5):
        pred_x, pred_y = kf_instance.predict()
        print(f"Predicted position: ({pred_x:.2f}, {pred_y:.2f})")

        meas_x = pred_x + np.random.normal(0, 0.1)
        meas_y = pred_y + np.random.normal(0, 0.1)

        est_x, est_y = kf_instance.update(meas_x, meas_y)
        print(f"Updated position: ({est_x:.2f}, {est_y:.2f})\n")