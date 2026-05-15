import os
import cv2
import numpy as np
import numba as nb


def callback(input):
    pass


def _clamp_block_size(v: int) -> int:
    return max(int(v), 2)


def _make_odd_at_least(v: int, min_v: int = 3) -> int:
    v = max(int(v), min_v)
    return v if (v % 2 == 1) else v + 1


@nb.njit(cache=True)
def find_points_above_threshold(cornerness: np.ndarray, th: float):
    rows, cols = cornerness.shape
    count = 0
    for r in range(rows):
        for c in range(cols):
            if cornerness[r, c] > th:
                count += 1

    ys = np.empty(count, dtype=np.int32)
    xs = np.empty(count, dtype=np.int32)

    k = 0
    for r in range(rows):
        for c in range(cols):
            if cornerness[r, c] > th:
                ys[k] = r
                xs[k] = c
                k += 1

    return ys, xs


def sift_detector():
    root = os.getcwd()
    img_path = os.path.join(root, 'data/dji_fly_20240919_171150_122_1726758725363_photo_optimized.jpg')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize image
    h, w, _ = img.shape
    scale_factor = 1 / 5
    heightScale = int(h * scale_factor)
    widthScale = int(w * scale_factor)
    img = cv2.resize(img, (widthScale, heightScale), interpolation=cv2.INTER_LINEAR)

    # apply Gause bluring  on image
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # house_gray = gray[1250:, 1000:2250].astype(np.uint8)
    # house_color = img[1250:, 1000:2250]

    house_gray = gray.copy()
    house_color = img.copy()

    win_name = 'sift_detector'
    cv2.namedWindow(win_name)

    # Initialize to valid defaults (avoid 0 and even ksize)
    cv2.createTrackbar("blockSize", win_name, 2, 80, callback)
    cv2.createTrackbar("ksize", win_name, 3, 31, callback)
    cv2.createTrackbar("th_coef", win_name, 10, 100, callback)  # percent of max

    while True:
        if cv2.waitKey(1) == ord('q'):
            break

        blockSize = _clamp_block_size(cv2.getTrackbarPos("blockSize", win_name))
        ksize = _make_odd_at_least(cv2.getTrackbarPos("ksize", win_name), 3)
        th_coef = cv2.getTrackbarPos("th_coef", win_name) / 100.0

        cornerness = cv2.cornerHarris(house_gray, blockSize=blockSize, ksize=ksize, k=0.04)
        cornerness[cornerness < 0] = 0
        cornerness = np.log(cornerness + 1e-6)

        th = float(th_coef * np.max(cornerness))
        ys, xs = find_points_above_threshold(cornerness, th)

        result = house_color.copy()
        for y, x in zip(ys, xs):
            cv2.circle(result, (int(x), int(y)), 14, (255, 0, 0), 2)

        cv2.imshow(win_name, result)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sift_detector()
