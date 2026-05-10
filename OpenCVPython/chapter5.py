# Warp perspective
import cv2
import numpy as np


def main():
    width, height = 250, 350
    pst1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]]) # square of original image
    pst2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) # points of aligned image
    img = cv2.imread('resources/cards.jpg')
    matrix = cv2.getPerspectiveTransform(pst1, pst2) # prepare matrix
    warpedImg = cv2.warpPerspective(img, matrix, (width, height)) # wrapping method

    cv2.imshow('original', img)
    cv2.imshow('warped', warpedImg)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
