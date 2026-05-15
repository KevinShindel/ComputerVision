import os

import cv2


def preprocessing(img, clipLimit=2.5):
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(image_hsv)
    clahe_value = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8)).apply(value)
    cl_luma = cv2.cvtColor(cv2.merge([hue, saturation, clahe_value]), cv2.COLOR_HSV2RGB)
    return cl_luma


def main():
    path_to_images = "../../data/coin_dataset/images"
    images = os.listdir(path_to_images)

    for n, i in enumerate(images):
        print(f"Total processed: {n} of {len(images)}")
        number = i.split(".")[0]
        annotation = f"../../data/coin_dataset/labels/{number}.txt"
        abs_img_path = os.path.join(path_to_images, i)
        img = cv2.imread(abs_img_path, cv2.IMREAD_COLOR)

        H, W = img.shape[:2]

        with open(annotation, "r") as handler:
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

                file_to_save = cv2.resize(
                    crop, (128, 128), interpolation=cv2.INTER_AREA
                )  # down scaling !
                # Preprocessing
                file_to_save = preprocessing(file_to_save)

                cls_folder = os.path.join(
                    "../../data/", "coin_tf_dataset", str(int(cls))
                )
                if not os.path.exists(cls_folder):
                    os.makedirs(cls_folder, True)

                cur_number = len(os.listdir(cls_folder)) + 1
                file_name = f"{cur_number}.jpg"
                path_to_save = os.path.join(cls_folder, file_name)
                cv2.imwrite(path_to_save, file_to_save)


if __name__ == "__main__":
    main()
