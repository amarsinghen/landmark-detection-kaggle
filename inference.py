import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
# Since there is no training, we want to run this code only on cpu save gpu resources
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

def simple_inference(uploaded_image_path):
    '''
    Doing simple inference for single uploaded image.
    :param uploaded_image_path: string, path to the uploaded image
    '''

    image = load_image(uploaded_image_path)
    predicted_class, confidence_score = predict_class_confidence_score(image)
    return confidence_score*100, predicted_class


def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image


def load_image(uploaded_image_path):
    image = resize_image(uploaded_image_path, 224, 224)
    np_image = np.array(image)
    np_image = np_image / 255.0
    image = np.expand_dims(np_image, axis=0)
    return image


def predict_class_confidence_score(image):
    loaded_models_list, indices_to_class_labels_dict_list = load_model_weights_and_class_labels()
    results_list = []
    predicted_class_list = []
    confidence_score_list = []
    max_confidence_score_tuple = (0, 0)
    for j in range(len(loaded_models_list)):
        results_list.append(loaded_models_list[j].predict(image))
        predicted_class_list.append(indices_to_class_labels_dict_list[j][np.argmax(results_list[j])])
        confidence_score_list.append(max(results_list[j][0]))
        logging.info("Identified landmark for image is : {}".format(predicted_class_list[j]))
        logging.info("Confidence score : {}".format(confidence_score_list[j]))
        logging.info("Whole Result array : {}\n".format(sorted(results_list[j][0])[-5:]))
        if confidence_score_list[j] > max_confidence_score_tuple[1]:
            max_confidence_score_tuple = (predicted_class_list[j], confidence_score_list[j])
    return max_confidence_score_tuple[0], max_confidence_score_tuple[1]


def load_model_weights_and_class_labels():
    num_of_models = 2
    loaded_models_list = []
    indices_to_class_labels_dict_list = []

    for i in range(num_of_models):
        filepath = "model_weights/group{}_set224_resnet50_weights.hdf5".format(i + 10)
        loaded_models_list.append(tf.keras.models.load_model(filepath))
        with open("model_weights/group{}_indices_to_class_labels_dict.json".format(i + 10), "rb") as pickle_file:
            indices_to_class_labels_dict_list.append(pd.read_pickle(pickle_file))
    return loaded_models_list, indices_to_class_labels_dict_list
