import cv2
import os

def callback(input):
    pass

def canny_edge():
    root = os.getcwd()
    img_path = os.path.join(root, 'data/dji_fly_20240919_171150_122_1726758725363_photo_optimized.jpg')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    scale = 1/5

    heightScale = int(h * scale)
    widthScale = int(w * scale)

    img = cv2.resize(img, (widthScale, heightScale), interpolation=cv2.INTER_LINEAR)

    win_name = 'canny'

    cv2.namedWindow(win_name)
    cv2.createTrackbar('min_thres', win_name,0,1000, callback)
    cv2.createTrackbar('max_thres', win_name,0,1000, callback)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break
        min_thres = cv2.getTrackbarPos('min_thres', win_name)
        max_thres = cv2.getTrackbarPos('max_thres', win_name)
        cannyEdge = cv2.Canny(img, min_thres, max_thres)
        cv2.imshow(win_name, cannyEdge)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    canny_edge()