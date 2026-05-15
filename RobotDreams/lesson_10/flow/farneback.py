import cv2
import numpy as np


def farneback_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                   pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                   iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2,
                   flags: int = 0) -> np.ndarray:
    """
    Returns dense optical flow (H x W x 2) from prev -> curr, where flow[...,0]=dx, flow[...,1]=dy.
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags
    )
    return flow


def flow_to_hsv_bgr(flow: np.ndarray, clip_mag: float | None = None) -> np.ndarray:
    """
    Visualize flow as HSV:
    Hue = direction, Value = magnitude.
    """
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    if clip_mag is not None:
        mag = np.clip(mag, 0.0, float(clip_mag))

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang * 180.0 / np.pi) / 2.0).astype(np.uint8)  # [0,180)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(bgr: np.ndarray, flow: np.ndarray, step: int = 16,
                     color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw a sparse arrow field over an image.
    """
    h, w = bgr.shape[:2]
    vis = bgr.copy()

    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].astype(int)
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]

    for (x0, y0, dx, dy) in zip(x.flatten(), y.flatten(), fx.flatten(), fy.flatten()):
        x1 = int(round(x0 + dx))
        y1 = int(round(y0 + dy))
        cv2.arrowedLine(vis, (int(x0), int(y0)), (x1, y1), color, 1, tipLength=0.3)

    return vis


# Example usage:
if __name__ == "__main__":
    path = '../../video/people_walking.mp4'
    cap = cv2.VideoCapture(path)  # or path to video file
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from camera/video.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        flow = farneback_flow(prev, frame)
        flow_vis = flow_to_hsv_bgr(flow)
        arrows = draw_flow_arrows(frame, flow, step=20)

        cv2.imshow("Frame", frame)
        cv2.imshow("Farneback HSV", flow_vis)
        cv2.imshow("Farneback Arrows", arrows)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        prev = frame

    cap.release()
    cv2.destroyAllWindows()
