import os
import shutil
import glob
import random

def move_images(args):
    list_of_images, imageIds_landmarkIds_dict = args[0], args[1]
    count=0
    for filepath in list_of_images:
        file_name = filepath.split("\\")[-1].split(".")[0]
        if (file_name in imageIds_landmarkIds_dict):
            count += 1
            landmark_id_folder_name = imageIds_landmarkIds_dict[file_name]
            local_location = str("datasets/raw_data/train/" + str(landmark_id_folder_name) + "/" + str(file_name) + ".jpg")
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
            shutil.move(filepath, local_location)
#             break
    return count

def copy_images(group_and_landmark):
    group_number, landmark_id_path = group_and_landmark[0], group_and_landmark[1]
    new_dir_location = landmark_id_path.replace("datasets", "datasets\\group{}_set_224".format(group_number))
    print(new_dir_location)
    if not os.path.isdir(new_dir_location):
        try:
            os.makedirs(new_dir_location)
        except:
            pass
    list_of_images = [os.path.join(landmark_id_path, image_file) for image_file in os.listdir(landmark_id_path)] 
    for image_path in list_of_images:
        shutil.copy2(image_path, new_dir_location)


def move_validation_images(args):
    landmark_id_path, num_of_images_to_move = args[0], args[1]
    new_dir_location = landmark_id_path.replace("train", "valid")
    if not os.path.isdir(new_dir_location):
        try:
            os.makedirs(new_dir_location)
        except:
            pass
    list_of_images_path = glob.glob("{}\\*.*".format(landmark_id_path))
    list_of_images_to_move = random.sample(list_of_images_path,num_of_images_to_move)
    for image_path in list_of_images_to_move:
        shutil.move(image_path, new_dir_location)