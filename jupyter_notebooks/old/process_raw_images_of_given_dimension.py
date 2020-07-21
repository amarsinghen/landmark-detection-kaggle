import os
import cv2
import numpy as np

def process_save_images(args):
    list_of_images, imageIds_landmarkIds_dict, dimension = args[0], args[1], args[2]
    count=0
    for filepath in list_of_images:
        file_name = filepath.split("\\")[-1].split(".")[0]
        if (file_name in imageIds_landmarkIds_dict):
            count += 1
            landmark_id_folder_name = imageIds_landmarkIds_dict[file_name]
            local_location = str("datasets/set_" + str(dimension[0]) + "/train/" + str(landmark_id_folder_name) + "/" + str(file_name) + ".jpg")
            if not os.path.isdir(local_location.split(str(file_name))[0]):
                try:
                    os.makedirs(local_location.split(str(file_name))[0])
                except:
                    pass
#             print("landmark_id_folder_name " + str(landmark_id_folder_name))
#             print("file_name " + str(file_name))
#             print("list of images " + str(list_of_images[:4]))
#             print("filepath " + str(filepath))
#             break
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            image = resize2SquareKeepingAspectRation(image, dimension[0])
            cv2.imwrite(local_location, image)
#             break
    return count

def resize2SquareKeepingAspectRation(img, size):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        return cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
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
    return cv2.resize(mask, (size, size), interpolation = cv2.INTER_AREA)
# # prnt(process_save_images(list_of_images))
