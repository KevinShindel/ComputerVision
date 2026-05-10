# joining images
import cv2
import numpy as np


def main():
    img = cv2.imread('resources/Lenna.png')
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    v_attached = np.vstack((img, img))
    h_attached = np.hstack((img, img))
    cv2.imshow('Vertical', v_attached)
    cv2.imshow('Horizontal', h_attached)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
