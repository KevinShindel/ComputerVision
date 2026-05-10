# Draw shapes and text
import cv2
import numpy as np


def main():
    img = np.zeros((512, 512, 3), np.uint8)
    img[:] = 255, 0, 0  # paint all picture in blue color

    img[200:300, 300:400] = 125, 125, 0  # paint piece of pictute

    cv2.line(img, (0, 0), (img.shape[0], img.shape[1]), (0, 255, 255), 1)  # draw line
    cv2.rectangle(img, (20, 20), (250, 300), (0, 0, 255), 3)  # draw rectangle
    cv2.circle(img, (256, 256), 30, (255, 255, 0), 2)  # draw circle
    cv2.putText(img, ' OpenCV ', (20, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)  # draw text

    cv2.imshow('Zeroes', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
