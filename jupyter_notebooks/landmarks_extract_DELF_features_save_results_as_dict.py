# This class only generates the delf features for all the images and store them in pickle gzip file

import numpy as np
from PIL import Image, ImageOps
import time
import glob
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
import pickle
import gc
import gzip

def main():
    
    delf = load_delf()
    
    #Loading list of image_ids to work with
    with open('landmark_ids_greater_than_5_list.json') as json_file:
        landmark_ids_greater_than_5_list = json.load(json_file)
    type(landmark_ids_greater_than_5_list)
    print(len(landmark_ids_greater_than_5_list))
    print(landmark_ids_greater_than_5_list[:5])
    
    start_time_extract_features_all_images = time.time()
    
#     landmark_ids_list = [0,1,2,3]
    for landmark_id in landmark_ids_greater_than_5_list:
        start_time_process_landmark_id = time.time()
        
        # creating dictionary of landmark_ids and delf features for all landmarks that has >5 images to later use to compare features
        landmark_ids_delf_features_dict = {}
    
        images_paths = glob.glob("datasets\\raw_data\\train\\{}\\*.jpg".format(landmark_id))

        images = []
        for i in range (0,len(images_paths)):
            images.append(download_and_resize(images_paths[i]))
       
        delf_results_list = get_deep_local_features(images, delf)

        landmark_ids_delf_features_dict[landmark_id] = convert_to_numpy_and_add_images_id(delf_results_list, images_paths)
        
         #Pickling the data: dictionary of landmarkids and delf features to be used in the feature matching script
        with gzip.open("landmark_ids_delf_features_dict_pickle_files\\{}.pkl.gz".format(landmark_id), "wb") as pickle_file:
            pickle.dump(landmark_ids_delf_features_dict, pickle_file)
        
#         with open("landmark_ids_delf_features_dict_pickle_files\\{}.pkl".format(landmark_id), "wb") as pickle_file:
#             pickle.dump(landmark_ids_delf_features_dict, pickle_file)
        
        print("landmark_id : {}, Num of Images : {}, Processing Time : {}".format(landmark_id, len(images), time.time()-start_time_process_landmark_id))
    
        del images_paths
        del images
        del delf_results_list
        del landmark_ids_delf_features_dict
        gc.collect()
    
    print("\nTotal time to extract delf features for all images in all landmarks: ", time.time() - start_time_extract_features_all_images)
    
    
# loading DELF
def load_delf():
    return hub.load('https://tfhub.dev/google/delf/1').signatures['default']

def download_and_resize(path, new_width = 256, new_height = 256):
    image = Image.open(path)
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image

def run_delf(image, delf):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)
    return delf(
        image=float_image,
        score_threshold=tf.constant(100.0),
        image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
        max_feature_num=tf.constant(1000))

def get_deep_local_features(images, delf):
    run_delf_image_list_results = []
    # start_time = time.time()
    for i in range (0,len(images)):
        run_delf_image_list_results.append(run_delf(images[i], delf))
    # print("(Images : ", len(images), ", time: ", time.time() - start_time, ")")
    return run_delf_image_list_results

# Converting the delf results to numpy. It is easy to store this way
# Also adding the image_ids to results list for individual images
# only adding image ids instead of whole image path, because it saves memory while pickling
def convert_to_numpy_and_add_images_id(results, images_paths):
    locations_descriptors_dict_list = []
    
    for i in range(len(results)):
        proto_tensor_result_locations = tf.make_tensor_proto(results[i]['locations'])
        proto_tensor_result_descriptors = tf.make_tensor_proto(results[i]['descriptors'])
        result_locations_ndarray = tf.make_ndarray(proto_tensor_result_locations)
        result_descriptors_ndarray = tf.make_ndarray(proto_tensor_result_descriptors)
        result_np_dict = {}
        result_np_dict['locations'] = result_locations_ndarray
        result_np_dict['descriptors'] = result_descriptors_ndarray
        result_np_dict['image_id'] = images_paths[i].split("\\")[-1].split(".jpg")[0]
        locations_descriptors_dict_list.append(result_np_dict)
    return locations_descriptors_dict_list 
    
# images_to_filter_out_after_delf.update(delf_compare_images_multiprocessing.compare_images(images_paths, images, results))
if __name__ == '__main__':
    main()