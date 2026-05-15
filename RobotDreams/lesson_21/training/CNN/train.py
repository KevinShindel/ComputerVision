import cv2


def main():
    image = "../../data/coin_dataset/images/025.jpg"
    label = "../../data/coin_dataset/labels/025.txt"

    img = cv2.imread(image, cv2.IMREAD_COLOR)

    H, W = img.shape[:2]

    with open(label, "r") as handler:
        for line in handler.readlines():
            cls, x, y, w, h = map(float, line.split())

            # convert
            x_center = x * W
            y_center = y * H
            box_width = w * W
            box_height = h * H

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            crop = img[y1:y2, x1:x2]
            cv2.imshow(str(cls), crop)

            # draw rectangle
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                str(cls),
                (x1 + 5, y1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

    cv2.imshow("Test", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
