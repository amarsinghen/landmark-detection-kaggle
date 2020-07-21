import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import time
import glob
from multiprocessing import Pool, Manager
import pickle
import json
import gc
import gzip
import os

def main():
    
    manager = Manager()
    
# Create a dict of images/files, max_inliers to filter out after DELF
    images_to_filter_out_after_delf_dict = manager.dict()
    
    pickle_files_dir = "landmark_ids_delf_features_dict_pickle_files"

    with open('landmark_ids_greater_than_5_list.json') as json_file:
        landmark_ids_greater_than_5_list = json.load(json_file)
    print(len(landmark_ids_greater_than_5_list))
    print(landmark_ids_greater_than_5_list[:5])
#     landmark_ids = landmark_ids_greater_than_5_list
    
    total_time_start = time.time()
#     landmark_ids = [0,1,2,3]
#     landmark_ids = [176528,177870]
    for landmark_id in landmark_ids_greater_than_5_list:
        counts = manager.Value('i',0)
        
        # Loading pickle file with landmark_ids and and their location and descriptors features
        with gzip.open("{}\\{}.pkl.gz".format(pickle_files_dir, landmark_id),'rb') as f:
             landmark_ids_results_dict = pickle.load(f)

        results = landmark_ids_results_dict[landmark_id]

        # the multiprocessing_results_list is of type list of tuples, each tuple has position 0 as dictionary, position 1 as list of dictionaries
        multiprocessing_results_list = []
        for i in range(0, len(results)-1):
            multiprocessing_results_list.append((results[i], results[i+1:], images_to_filter_out_after_delf_dict, landmark_id, counts))

        time_to_compare_all_images_in_class = time.time()
        
        compare_images_multiprocessing(multiprocessing_results_list)
                
        print("landmark_id : {}, time : {}, Num of non-landmark images :  {}".format(str(landmark_id), time.time() - time_to_compare_all_images_in_class, counts.value))
                
        del results
        del multiprocessing_results_list
        gc.collect()
        
    print("\n")
    print(len(images_to_filter_out_after_delf_dict))
#     print("\n")
    print("Total time to compare all images : {}".format(time.time()-total_time_start))
    
    # Creating json file to save info of all the images that needs to be deleted to clean the data for training
    images_to_filter_out_after_delf_json = json.dumps(images_to_filter_out_after_delf_dict.copy())
    file = open("images_to_filter_out_after_delf_dict.json","w")
    file.write(images_to_filter_out_after_delf_json)
    file.close()

def compare_images_multiprocessing(multiprocessing_results_list):
    p = Pool()
    p.map(match_images3, (multiprocessing_results_list))
    p.close()
    p.join()    
#     return inliers

#@title TensorFlow is not needed for this post-processing and visualization
def match_images3(multiprocessing_results_list_item):

    train_images_path = "datasets\\raw_data\\train"
    
    result1, image_id, images_to_filter_out_after_delf_dict, landmark_id, counts = multiprocessing_results_list_item[0], multiprocessing_results_list_item[0]['image_id'], multiprocessing_results_list_item[2], multiprocessing_results_list_item[3], multiprocessing_results_list_item[4]
    
    list_results = multiprocessing_results_list_item[1]

    max_inliers_reached = 10
    max_inlier = 0
    for j in range(len(list_results)):
        result2 = list_results[j]
            
        distance_threshold = 0.8
        # Read features.
        num_features_1 = result1['locations'].shape[0]
        num_features_2 = result2['locations'].shape[0]

        # Find nearest-neighbor matches using a KD tree.
        d1_tree = cKDTree(result1['descriptors'])
        _, indices = d1_tree.query(
            result2['descriptors'],
            distance_upper_bound=distance_threshold)

        # Select feature locations for putative matches.
        locations_2_to_use = np.array([
            result2['locations'][i,]
            for i in range(num_features_2)
            if indices[i] != num_features_1
        ])

        locations_1_to_use = np.array([
            result1['locations'][indices[i],]
            for i in range(num_features_2)
            if indices[i] != num_features_1
        ])

        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
                (locations_1_to_use, locations_2_to_use),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)

            sum_inliers = sum(inliers)
        except:
            sum_inliers = 0
            
#         if num_features_1 != sum_inliers:
        if max_inlier < sum_inliers:
            max_inlier = sum_inliers
        
        if sum_inliers > max_inliers_reached:
            break
        
    if max_inlier <= max_inliers_reached:
        images_to_filter_out_after_delf_dict["{}.jpg".format(os.path.join(train_images_path, str(landmark_id), image_id))] = int(max_inlier)
#         list_to_skip.append(image_id)
        counts.value += 1
    
if __name__ == '__main__':
    main()
