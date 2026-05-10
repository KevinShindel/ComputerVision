# Intro CV2
import cv2 as cv


def read_img():
    img = cv.imread('resources/Lenna.png')
    cv.imshow('Output', img)
    cv.waitKey(0)


def video_capture():
    cap = cv.VideoCapture('resources/test_video.mp4')
    while True:
        success, img = cap.read()
        cv.imshow('Video', img)
        if cv.waitKey(1) & 0xff == ord('q'):
            break


def camera_capture():
    cap = cv.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, img = cap.read()
        cv.imshow('Video', img)
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    # Destroy all the windows
    cv.destroyAllWindows()


if __name__ == '__main__':
    camera_capture()
