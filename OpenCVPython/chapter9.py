# face recognition
import cv2


def main():
    face_cascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
    img = cv2.imread('resources/Lenna.png')

    img = cv2.resize(img, (400, 300))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()
