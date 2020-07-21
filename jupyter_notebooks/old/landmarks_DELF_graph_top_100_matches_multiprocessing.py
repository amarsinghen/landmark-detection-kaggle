import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import time
import glob
from multiprocessing import Pool
import pickle
import json
import gc


def main():
# Create a dict of images/files, max_inliers to filter out after DELF
    images_to_filter_out_after_delf_dict = {}
    
    # Loading pickle file with landmark_ids and and their location and descriptors features
    filename = 'landmark_ids_images_delf_features_dict1.pkl'
    with open(filename,'rb') as f:
         landmark_ids_results_dict = pickle.load(f)
    print(len(landmark_ids_results_dict))
    landmark_ids = list(landmark_ids_results_dict.keys())
        
#     landmark_ids = [0,1,2,3]
    for landmark_id in landmark_ids:
        images_paths = glob.glob("datasets\\raw_data\\train\\" + str(landmark_id) + "\\*.jpg")
#         print("Number of images in the landmark_id, ", landmark_id ," : ", len(images_paths))
#         print(images_paths[:2])

        results = landmark_ids_results_dict[landmark_id]

        # the multiprocessing_results_list is of type list of tuples, each tuple has position 0 as dictionary, position 1 as list of dictionaries
        multiprocessing_results_list = []
        for i in range(0, len(results)):
            multiprocessing_results_list.append((results[i], results, images_paths[i]))

        time_to_compare_all_images_in_class = time.time()
        
        # below it is a list of tuples (image_path_i, max_inliers, location_j of 2nd image which we can find from images_paths)
        imagePath_maxInliers_tuple_list = compare_images_multiprocessing(multiprocessing_results_list)

        count = 0
        for tuple_item in imagePath_maxInliers_tuple_list:
            if tuple_item[1] <= 10:
                count += 1
                images_to_filter_out_after_delf_dict[tuple_item[0]] = int(tuple_item[1])
                
        print(" images : {}, landmark_id : {}, time : {}, non-landmark images :  {}".format(str(len(images_paths)), str(landmark_id), time.time() - time_to_compare_all_images_in_class, count ))
                
        del results
        del multiprocessing_results_list
        del imagePath_maxInliers_tuple_list
        del images_paths
        gc.collect()
        
    print("\n")
    print(len(images_to_filter_out_after_delf_dict))
    print("\n")
    print(images_to_filter_out_after_delf_dict)
    
    # Creating json file to save info of all the images that needs to be deleted to clean the data for training
    images_to_filter_out_after_delf_json = json.dumps(images_to_filter_out_after_delf_dict)
    file = open("images_to_filter_out_after_delf_dict.json","w")
    file.write(images_to_filter_out_after_delf_json)
    file.close()

def compare_images_multiprocessing(multiprocessing_results_list):
    p = Pool(4)
    inliers = p.map(match_images3, (multiprocessing_results_list))
    p.close()
    p.join()    
    return inliers

#@title TensorFlow is not needed for this post-processing and visualization
def match_images3(multiprocessing_results_list_item):

    result1, image_path = multiprocessing_results_list_item[0], multiprocessing_results_list_item[2]
    list_results = multiprocessing_results_list_item[1]

    location_j = 0
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
            
        if num_features_1 != sum_inliers:
            if max_inlier < sum_inliers:
                max_inlier = sum_inliers
                location_j = j
    
    return (image_path, max_inlier, location_j)
    
if __name__ == '__main__':
    main()
