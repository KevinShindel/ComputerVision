# Contours and shape detection
import cv2
import numpy as np

from common import stack_images


def get_contours(img, img_contour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor == 3:
                object_type = "Tri"
            elif obj_cor == 4:
                asp_ratio = w / float(h)
                if 0.98 < asp_ratio < 1.03:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif obj_cor > 4:
                object_type = "Circles"
            else:
                object_type = "None"

            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_contour, object_type,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)
    return img_contour


def main():

    path = 'Resources/shapes.png'
    img = cv2.imread(path)
    img_contour = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    img_canny = get_contours(img_canny, img_contour)

    img_blank = np.zeros_like(img)
    imgStack = stack_images(0.8, ([img, img_gray, img_blur],
                                  [img_canny, img_contour, img_blank]))

    cv2.imshow("Stack", imgStack)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
