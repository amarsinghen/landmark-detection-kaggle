import cv2
import numpy as np

def process_save_images(image_path):
    local_location = image_path.replace("raw_data", "set_256")
    # print(image_path)
    # print("local location1 : {}".format(local_location1))
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = resize2SquareKeepingAspectRation(image, 256)
    cv2.imwrite(local_location, image)


def resize2SquareKeepingAspectRation(img, size):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        if h < size or w < size:
            return cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
        else:
            return cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    if h < size or w < size:
        return cv2.resize(mask, (size, size), interpolation = cv2.INTER_AREA)
    else:
        return cv2.resize(mask, (size, size), interpolation = cv2.INTER_CUBIC)
# # prnt(process_save_images(list_of_images))
