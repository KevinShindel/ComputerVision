# Basic functions
import cv2
import numpy as np

def resize_and_blur():
    img = cv2.imread('resources/Lenna.png')

    print('Original Dimensions : ', img.shape)

    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    kernel = np.ones((5,5), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 200, 200)
    imgDialation = cv2.dilate(imgCanny, kernel, 1)
    imgEroded = cv2.erode(imgDialation, kernel, 1)


    cv2.imshow('Gray image', imgGray)
    cv2.imshow('Blur image', imgBlur)
    cv2.imshow('Canny image', imgCanny)
    cv2.imshow('Dialation image', imgDialation)
    cv2.imshow('Erroded image', imgEroded)

    cv2.waitKey(0)


if __name__ == '__main__':
    resize_and_blur()
