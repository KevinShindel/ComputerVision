import os

import cv2


def load_lena():
    root = os.getcwd()
    path = os.path.join(root, '../../images/Lenna.png')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

