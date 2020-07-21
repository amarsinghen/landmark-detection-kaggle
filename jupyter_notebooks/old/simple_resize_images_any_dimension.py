import os
import cv2

def process_save_images(list_of_images, landmark_id, dimension):
    # create landmark labeled directory if it does not exists
    directory_location = "datasets/set_" + str(dimension[0]) + "/train/" + str(landmark_id)
    if not os.path.isdir(directory_location):
        try:
            os.makedirs(directory_location)
        except:
            pass

    for filepath in list_of_images:
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
        cv2.imwrite(str(directory_location + "/" + str(filepath.split("\\")[1].split(".")[0]) + ".jpg"), image)
    
# print(process_save_images(list_of_images))
