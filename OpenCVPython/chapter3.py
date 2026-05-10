# Resizing and cropping
import cv2


def main():
    img = cv2.imread('resources/lambo.png')
    print(img.shape)
    imgResize = cv2.resize(img, (300, 200))   # Resizing picture

    imgCropped = img[0:200, 200:400]  # cropping via coordinates

    cv2.imshow('Original', img)
    cv2.imshow('Resized', imgResize)
    cv2.imshow('Cropped', imgCropped)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
