import os
import cv2
import time

def process_save_images(args):
    list_of_images, imageIds_landmarkIds_dict = args[0], args[1]
    count=0
    for filepath in list_of_images:
        file_name = filepath.split("\\")[-1].split(".")[0]
        if (file_name in imageIds_landmarkIds_dict):
            count += 1
            landmark_id_folder_name = imageIds_landmarkIds_dict[file_name]
            local_location = str("tmp/train/" + str(landmark_id_folder_name) + "/" + str(file_name) + ".jpg")
            if not os.path.isdir(local_location.split(str(file_name))[0]):
                try:
                    os.makedirs(local_location.split(str(file_name))[0])
                except:
                    pass
#             print(len(landmark_id))
#             print(str(landmark_id_folder_name))
#             print(file_name)
            image = cv2.imread(filepath)
            image = cv2.resize(image, (448,448))
            cv2.imwrite(local_location, image)
    return count
# print(process_save_images(list_of_images))
